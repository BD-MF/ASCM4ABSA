# -*- coding: utf-8 -*-

import os
import torch
import shutil
import random
import itertools
import numpy as np
import torch.nn as nn
from logger import Logger
from sklearn import metrics
import torch.nn.functional as F
from data_loader import ASCDataLoader
from transformers import AutoTokenizer, AutoModel, AutoConfig

class Evaluator:
    def __init__(self, args):
        self.logger = Logger(os.path.join(args.exp_dir, '{}_result.txt'.format(args.test_data_name)))
        self._print_args(args)

    def _print_args(self, args):
        print('>> training arguments:')
        for arg in vars(args):
            print('>> {}: {}'.format(arg, getattr(args, arg)))
    
    def output_fn(self,output, target, mask):
        batch_size = mask.shape[0]
        output_container, target_container = [], []
        for i in range(batch_size):
            output_container.append(torch.masked_select(output[i].argmax(dim=-1), mask[i].bool()).cpu().numpy().tolist())
            target_container.append(torch.masked_select(target[i], mask[i].bool()).cpu().numpy().tolist())
        return output_container, target_container



    def score_BIO(self, golden, predicted, ignore_index=-1):
        # 'B': 1, 'I': 2, 'O': 0
        assert len(predicted) == len(golden)
        sum_all = 0
        sum_correct = 0
        golden_01_count = 0
        predict_01_count = 0
        correct_01_count = 0
        
        for i in range(len(golden)):
            length = len(golden[i])
            
            golden_01 = 0
            correct_01 = 0
            predict_01 = 0
            predict_items = []
            golden_items = []
            golden_seq = []
            predict_seq = []
            for j in range(length):
                if golden[i][j] == ignore_index:
                    break
                if golden[i][j] == 1:
                    if len(golden_seq) > 0:  # 00
                        golden_items.append(golden_seq)
                        golden_seq = []
                    golden_seq.append(j)
                elif golden[i][j] == 2:
                    if len(golden_seq) > 0:
                        golden_seq.append(j)
                elif golden[i][j] == 0:
                    if len(golden_seq) > 0:
                        golden_items.append(golden_seq)
                        golden_seq = []
                if predicted[i][j] == 1:
                    if len(predict_seq) > 0:  # 00
                        predict_items.append(predict_seq)
                        predict_seq = []
                    predict_seq.append(j)
                elif predicted[i][j] == 2:
                    if len(predict_seq) > 0:
                        predict_seq.append(j)
                elif predicted[i][j] == 0:
                    if len(predict_seq) > 0:
                        predict_items.append(predict_seq)
                        predict_seq = []
            if len(golden_seq) > 0:
                golden_items.append(golden_seq)
            if len(predict_seq) > 0:
                predict_items.append(predict_seq)
            golden_01 = len(golden_items)
            predict_01 = len(predict_items)
            correct_01 = sum([item in golden_items for item in predict_items])
            golden_01_count += golden_01
            predict_01_count += predict_01
            correct_01_count += correct_01
        precision = correct_01_count/predict_01_count if predict_01_count > 0 else 0
        recall = correct_01_count/golden_01_count if golden_01_count > 0 else 0
        f1 = 2*precision*recall/(precision +recall) if (precision + recall) > 0 else 0
        
        return   f1, precision, recall
    
    def evaluation_score(self, golden, predicted):
        assert len(golden) == len(predicted)
        TP = 0
        total_of_label = 0
        total_of_predict = 0
        n_samples = len(golden)
        
        for i in range(n_samples):
            # B:0, I:1, O:2
            for predict, target in zip(predicted[i], golden[i]):
                if predict not in [2]:
                    total_of_predict += 1
                
                if target not in [2]:
                    total_of_label += 1
                    if predict == target:
                        TP += 1
                    
        if total_of_label != 0:
            precision = TP / total_of_label
        else:
            precision = 0
        if total_of_predict != 0:
            recall = TP / total_of_predict
        else:
            recall = 0

        if precision > 0 or recall > 0:
            f1 = 2 * precision * recall / (precision + recall + 1e-5)
        else:
            f1 = 0
        
        return f1, precision, recall  

    def rescale(self, input_list):
        the_array = np.asarray(torch.squeeze(input_list).cpu().detach().numpy())
        the_max = np.max(the_array)
        the_min = np.min(the_array)
        rescale = (the_array - the_min) / (the_max - the_min) * 100
        return rescale.tolist()
    
    def calculate_attentions(self, attentions):
        """
        Calculates the attention weights by:
            1.Take the attention weights from the last multi-head attention layer assigned to the last_hidden_state [batch_size, seq_length, hidden_size]
            2.Average each token across attention heads
            3.Normalize across tokens
        :param attentions: list of dictionaries of the form
            {'layer_name':(batch_size,num_multihead_attn,sequence_length,seq_length)}
        :return: a tensor of weights [batch_size, seq_length]
        """
        
        last_multihead_attn = attentions[-1] #[32,12,128,128]
        #For each multihead attention, get the attention weights going into the all sequence tokens
        #Average across attention heads
        average_head_attn = torch.mean(last_multihead_attn, dim=1)#[batch_size,sequence_length,sequence_length] [32,128,128]
        average_all_sequence_attn = torch.mean(average_head_attn, dim=-2) #[32,128]
        average_attn = torch.squeeze(average_all_sequence_attn)
        return average_attn#[32,128]



    def _evaluate(self, args, model, data_loader, tokenizer):
        # switch model to evaluation mode
        model.eval()
        output_all, target_all = [], []
        with torch.no_grad():
            for t_batch in data_loader:
                t_inputs = [
                    t_batch[col].to(args.device) for col in args.input_fields
                ]
                target = t_batch['label_tag_list'].to(args.device)
                label_tag_mask = t_batch['label_tag_mask'].to(args.device)
                
                output = model(t_inputs) 
               
                output, target=self.output_fn(output, target, label_tag_mask)
                output_all.extend(output)
                target_all.extend(target)
        f1, precision, recall = self.score_BIO(golden= target_all, predicted = output_all)

        return   f1, precision, recall
       
    def run(self, args, embedding, test_data, tokenizer):
        self.logger('+' * 30 + ' evaluation on {} '.format(args.test_data_name) + '+' * 30)

        result_dict = {'precision': [], 'recall':[], 'f1': []}
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
            test_f1, test_precision, test_recall = self._evaluate(args, model, test_data_loader, tokenizer)
            
            self.logger('test precision: {:.4f}, test recall: {:.4f}, test f1: {:.4f}'.format(test_precision, test_recall, test_f1))
            result_dict['precision'].append(test_precision)
            result_dict['recall'].append(test_recall)
            result_dict['f1'].append(test_f1)
        self.logger('#' * 30 + ' average ' + '#' * 30)
        self.logger('precision: {:.4f}, recall: {:.4f}, f1: {:.4f}'.format(
            np.mean(result_dict['precision']), np.mean(result_dict['recall']), np.mean(result_dict['f1'])))
       
