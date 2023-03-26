# -*- coding: utf-8 -*-

import os
import time
import math
import torch
import random
import itertools
import numpy as np
import torch.nn as nn
from sklearn import metrics
import torch.nn.functional as F
from data_loader import ASCDataLoader
from transformers import AdamW, get_linear_schedule_with_warmup, BertModel,RobertaModel

class Trainer:
    def __init__(self, args):
        self._print_args(args)
        self.args = args
    def _print_args(self, args):
        print('>> training arguments:')
        for arg in vars(args):
            print('>> {}: {}'.format(arg, getattr(args, arg)))
    
    def loss_fn(self,output, target, mask):
        loss = F.cross_entropy(output.flatten(0, 1), target.flatten(0, 1), reduction='none', ignore_index= -1)
        #loss = F.nll_loss(output.flatten(0, 1), target.flatten(0, 1), reduction='none', ignore_index= -1)
        loss = torch.masked_select(loss, mask.flatten(0, 1).bool())
        loss = loss.sum() / mask.sum()
        return loss

    def _train(self, args, model, optimizer, scheduler, train_data_loader, dev_data_loader, state_dict_path):
        best_dev_f1 = 0
        best_dev_epoch = 0
        iter_step = 0
        for epoch in range(args.num_train_epochs):
            print('>' * 30 + 'epoch {}'.format(epoch + 1) + '>' * 30)
            for batch in train_data_loader:
                iter_step += 1
                
                model.train()
                optimizer.zero_grad()

                inputs = [
                    batch[col].to(args.device)
                    for col in args.input_fields
                ]
                target = batch['label_tag_list'].to(args.device)
                label_tag_mask = batch['label_tag_mask'].to(args.device)

                output = model(inputs)
                
                loss = self.loss_fn(output, target, label_tag_mask)
                loss.backward()
                optimizer.step()

                if scheduler is not None:
                    scheduler.step()

                if iter_step % args.log_interval == 0:
                    dev_f1, dev_precision, dev_recall = self._evaluate(args, model, dev_data_loader)
                    
                    print('train loss: {:.4f}, dev precision: {:.4f}, dev recall: {:.4f},  dev f1: {:.4f}'.
                        format(loss.item(), dev_precision, dev_recall, dev_f1))
                    
                    if dev_f1 > best_dev_f1:
                        print('>> new best model.')
                        best_dev_epoch = epoch
                        best_dev_f1 = dev_f1
                        torch.save(model.state_dict(), state_dict_path)
            
            if epoch - best_dev_epoch >= args.num_patience_epochs:
                print('>> early stop.')
                break
        return  best_dev_f1

    def output_fn(self,output, target, mask):
        batch_size = mask.shape[0]

        loss_container, output_container, target_container = [], [], []
        for i in range(batch_size):
            loss_i = F.cross_entropy(output[i], target[i], reduction='none', ignore_index= -1)
            loss_container.append(torch.masked_select(loss_i, mask[i].bool()).cpu().numpy().tolist())
            output_container.append(torch.masked_select(output[i].argmax(dim=-1), mask[i].bool()).cpu().numpy().tolist())
            target_container.append(torch.masked_select(target[i], mask[i].bool()).cpu().numpy().tolist())
            
        return loss_container, output_container, target_container



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
        # B:0, I:1, O:2
        assert len(golden) == len(predicted)
        TP = 0
        total_of_label = 0
        total_of_predict = 0
        n_samples = len(golden)
        
        for i in range(n_samples):
           
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


    def _evaluate(self, args, model, data_loader):
        model.eval()
        loss_all, output_all, target_all = [], [], []
        with torch.no_grad():
            for t_batch in data_loader:
                t_inputs = [
                    t_batch[col].to(args.device) for col in args.input_fields
                ]
                target = t_batch['label_tag_list'].to(args.device)
                label_tag_mask = t_batch['label_tag_mask'].to(args.device)
                output = model(t_inputs)
                
                loss, output, target=self.output_fn(output, target, label_tag_mask)
                
                loss_all.extend(loss)
                output_all.extend(output)
                target_all.extend(target)
        
        f1, precision, recall = self.score_BIO(golden= target_all, predicted = output_all)

        return   f1, precision, recall
               

    def run(self, args, embedding, train_data, dev_data):
        print('+' * 30 + ' training on {} '.format(args.train_data_name) + '+' * 30)
        for i in range(args.num_repeats):
            print('#' * 30 + ' repeat {} '.format(i + 1) + '#' * 30)

            train_data_loader = ASCDataLoader(
                train_data,
                batch_size=args.batch_size,
                shuffle=True
            )
            dev_data_loader = ASCDataLoader(
                dev_data, 
                batch_size=args.batch_size, 
                shuffle=False
            )

            model = args.model_class(args, embedding).to(args.device)
            
            temp_best_path = os.path.join(args.exp_dir, 'best_ckpt_{}.pt'.format(i))

            if 'bert' in args.model_name:
                no_decay = ['bias', 'LayerNorm.weight']
                grouped_parameters = [
                    {
                        'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                        'weight_decay': args.weight_decay,
                    },
                    {
                        'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 
                        'weight_decay': 0.0
                    },
                ]
                optimizer = AdamW(grouped_parameters, lr=args.learning_rate)
                scheduler = get_linear_schedule_with_warmup(optimizer, int(0.05 * args.num_train_epochs * len(train_data_loader)), args.num_train_epochs * len(train_data_loader))
            else:
                optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
                scheduler = None
            
            self._train(args, model, optimizer, scheduler, train_data_loader, dev_data_loader, temp_best_path)

