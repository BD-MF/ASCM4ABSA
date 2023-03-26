# -*- coding: utf-8 -*-

import os
import torch
import shutil
import random
import numpy as np
import torch.nn as nn
from logger import Logger
from sklearn import metrics
import torch.nn.functional as F
from data_loader import ASCDataLoader

class Evaluator:
    def __init__(self, args):
        self.logger = Logger(os.path.join(args.exp_dir, '{}_result.txt'.format(args.test_data_name)))
        self._print_args(args)

    def _print_args(self, args):
        print('>> training arguments:')
        for arg in vars(args):
            print('>> {}: {}'.format(arg, getattr(args, arg)))

    def _evaluate(self, args, model, data_loader):
        # switch model to evaluation mode
        model.eval()
        t_target_all, t_output_all = None, None
        with torch.no_grad():
            for t_batch in data_loader:
                t_inputs = [
                    t_batch[col].to(args.device) for col in args.input_fields
                ]
                t_target = t_batch['polarity'].to(args.device)
                t_output = model(t_inputs)
                if t_target_all is None:
                    t_target_all = t_target
                    t_output_all = t_output
                else:
                    t_target_all = torch.cat((t_target_all, t_target), dim=0)
                    t_output_all = torch.cat((t_output_all, t_output), dim=0)
        acc = metrics.accuracy_score(t_target_all.cpu().numpy(),
                                     torch.argmax(t_output_all, -1).cpu().numpy())
        f1 = metrics.f1_score(t_target_all.cpu().numpy(),
                              torch.argmax(t_output_all, -1).cpu().numpy(),
                              labels=[0, 1, 2],
                              average='macro')
        return acc, f1

    def run(self, args, embedding, test_data):
        self.logger('+' * 30 + ' evaluation on {} '.format(args.test_data_name) + '+' * 30)
        result_dict = {'acc': [], 'f1': []}
        for i in range(args.num_repeats):
            self.logger('#' * 30 + ' repeat {} '.format(i + 1) + '#' * 30)
            test_data_loader = ASCDataLoader(
                test_data,
                batch_size=args.batch_size,
                shuffle=False
            )
            model = args.model_class(args, embedding).to(args.device)
            temp_best_path = os.path.join(args.exp_dir, 'best_ckpt_{}.pt'.format(i))
            state_dict_wo_embed = torch.load(temp_best_path)
            if 'bert' not in args.model_name:
                state_dict_wo_embed.pop('embed.weight')
            model.load_state_dict(state_dict_wo_embed, strict=False)
            test_acc, test_f1 = self._evaluate(args, model, test_data_loader)
            self.logger('test acc: {:.4f}, test f1: {:.4f}'.format(test_acc, test_f1))
            
            result_dict['acc'].append(test_acc)
            result_dict['f1'].append(test_f1)
        self.logger('#' * 30 + ' average ' + '#' * 30)
        self.logger('acc: {:.4f}, f1: {:.4f}'.format(np.mean(result_dict['acc']), np.mean(result_dict['f1'])))
        
