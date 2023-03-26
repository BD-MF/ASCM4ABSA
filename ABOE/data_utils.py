# -*- coding: utf-8 -*-

import os
import json
import nltk
import spacy
import random
import pickle
import codecs
import numpy as np
from logger import Logger
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModel, AutoConfig


def build_embedding_for_bert(data_dir, cache_dir='cahces'):
    config = AutoConfig.from_pretrained(data_dir, cache_dir=cache_dir)                           
    embedding = AutoModel.from_pretrained(data_dir, config=config, cache_dir=cache_dir)
    return embedding


def build_tokenizer_for_bert(data_dir, cache_dir='caches', use_fast=True):
    tokenizer = AutoTokenizer.from_pretrained(data_dir, cache_dir=cache_dir, use_fast=True)
    return tokenizer


def towe_truncate_and_pad(indices, max_length=128, pad_idx=0):
    if len(indices) > max_length:
        print("len(indices) > max_length")
        indices = indices[:max_length-1] + [indices[-1]]

    _len = len(indices)
    indices = indices + [pad_idx] * (max_length - _len)
    mask = [1] * _len + [0] * (max_length - _len)
    
    return indices, mask

def tokenize_data_for_bert(text_token_list, target_tag_list, label_tag_list, tokenizer, max_length=128, token_type="before_blank"):
    label_list = [ 'O', 'B', 'I']
    #'B': 1, 'I': 2, 'O': 0
    tag_map = {}
    for i in range(len(label_list)):
        tag_map[label_list[i]] = i 
    
    return_data = {}
    
    token_map = {}
    text_tokens = []
    for i, token in enumerate(text_token_list):
        token_pieces = tokenizer.tokenize(token)
        token_map[i] = (len(text_tokens), len(text_tokens) + len(token_pieces)-1)
        text_tokens.extend(token_pieces)

    new_target_tag_list = []
    new_label_tag_list = []
    
    for key, value in token_map.items():
        token_spice_num = value[1] - value[0] +1
        for _ in range(token_spice_num):
            new_target_tag_list.append(tag_map[target_tag_list[key]])
            new_label_tag_list.append(tag_map[label_tag_list[key]])
    
    text_tokens.insert(0, tokenizer.cls_token)
    text_tokens.append(tokenizer.sep_token)
    
    new_target_tag_list.insert(0, tag_map['O'])
    new_target_tag_list.append(tag_map['O'])
    
    target_start = 0
    target_end = 0
    
    for loc, tag_type in  enumerate(new_target_tag_list):
        if tag_type == tag_map['B']: 
            target_start = loc
            target_end = loc
            while(target_end<len(new_target_tag_list)):
                if new_target_tag_list[target_end] != tag_map['O']:
                    target_end += 1
                else:
                    break
    target_boundary_indices = [target_start, target_end]
    
    new_label_tag_list.insert(0, tag_map['O'])
    new_label_tag_list.append(tag_map['O'])
    
    text_token_indices = tokenizer.convert_tokens_to_ids(text_tokens)
    
    text_token_indices, text_tokens_mask = towe_truncate_and_pad(text_token_indices, max_length=max_length,  pad_idx=tokenizer.pad_token_id)
    new_target_tag_list, target_tag_mask = towe_truncate_and_pad(new_target_tag_list, max_length=max_length,  pad_idx=-1)
    new_label_tag_list, label_tag_mask = towe_truncate_and_pad(new_label_tag_list, max_length=max_length,  pad_idx=-1)
    
    label_tag_mask[sum(label_tag_mask)-1] = 0
    label_tag_mask[0] = 0

    target_tag_mask[sum(target_tag_mask)-1] = 0
    target_tag_mask[0] = 0
    
    return_data ={
        'text_token_indices': text_token_indices,
        'text_tokens_mask':text_tokens_mask,
        'target_tag_list':new_target_tag_list,
        'target_tag_mask':target_tag_mask,
        'label_tag_list':new_label_tag_list,
        'label_tag_mask':label_tag_mask,
        'target_boundary_indices':target_boundary_indices
    }

    return return_data

def build_data_for_bert(data_dir, tokenizer, max_length=128,token_type="before_blank", bert_type="roberta"):
    data_dict = {'train': [], 'dev': [], 'test': []}
    set_types = ['train', 'test']
    label_list = ['O', 'B', 'I'] #0:O  1:B   2:I 
    
    tag_map = {}
    for i in range(len(label_list)):
        tag_map[label_list[i]] = i

    for set_type in set_types:
        
        text_tokens_list = []
        target_list = []
        text_target_list = []
        label_list = []

        fname = os.path.join(data_dir, '{}.tsv'.format(set_type))
        with codecs.open(fname, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i ==0:
                    continue
                s_id, sentence, target_tags, opinion_words_tags = line.split('\t')
                sentence_tokens = sentence.strip().split()
                
                prompt_sentence = "The target aspect is "
                prompt_sentence_tokens = prompt_sentence.strip().split()

                for i, token in enumerate(prompt_sentence_tokens):
                    prompt_sentence_tokens[i] =  ' ' + token

                w_t = target_tags.strip().split(' ')
                target = []
                target_start = 0
                target_end = 0
                for loc,t in enumerate(w_t):
                    t_ = t.split('\\')[-1]
                    if t_ == 'B':
                        target_start = loc
                        target_end = loc
                        while(target_end<len(w_t)):
                            if w_t[target_end].split('\\')[-1] != 'O':
                                target_end += 1
                            else:
                                break
                    target.append(t_)
                
                w_l = opinion_words_tags.strip().split(' ')
                label = []
                label_start = 0
                label_end = 0
                for loc, l in enumerate(w_l):
                    l_ = l.split('\\')[-1]
                    if l_ == 'B':
                        label_start = loc
                        label_end = loc
                        while(label_end<len(w_l)):
                            if w_l[label_end].split('\\')[-1] != 'O':
                                label_end += 1
                            else:
                                break
                    label.append(l_)

                
                
                for i, token in enumerate(sentence_tokens):
                    if i !=0:
                        token = ' ' + token
                    sentence_tokens[i] = token
                

                if token_type == "before_blank":
                    if bert_type == "roberta":
                        sentence_tokens.insert(target_start,' ')
                    else:
                        sentence_tokens.insert(target_start,'##')
                    target.insert(target_start, 'O')
                    label.insert(target_start, 'O')
                elif token_type == "after_blank":
                    if bert_type == "roberta":
                        sentence_tokens.insert(target_end, ' ')
                    else:
                        sentence_tokens.insert(target_end, '##')
                    target.insert(target_end, 'O')
                    label.insert(target_end, 'O')
                elif token_type == "double_blank":
                    if bert_type == "roberta":
                        sentence_tokens.insert(target_start,' ')
                        sentence_tokens.insert(target_end + 1, ' ')
                    else:
                        sentence_tokens.insert(target_start,'##')
                        sentence_tokens.insert(target_end + 1, '##')

                    target.insert(target_start,'O')
                    target.insert(target_end + 1, 'O')

                    label.insert(target_start,'O')
                    label.insert(target_end + 1, 'O')
                elif token_type == "before_token":
                    sentence_tokens.insert(target_start,'<asp>')
                    target.insert(target_start,'O')
                    label.insert(target_start,'O')
                elif token_type == "after_token":
                    sentence_tokens.insert(target_end, '</asp>')
                    target.insert(target_end, 'O')
                    label.insert(target_end, 'O')
                elif token_type == "double_token":
                    sentence_tokens.insert(target_start,'<asp>')
                    sentence_tokens.insert(target_end + 1, '</asp>')

                    target.insert(target_start,'O')
                    target.insert(target_end + 1, 'O')

                    label.insert(target_start,'O')
                    label.insert(target_end + 1, 'O')
                    
                elif token_type == "sentence_pair_first":
                    sentence_tokens.append(tokenizer.sep_token)
                    target.append('O')
                    label.append('O')
                    aspects = sentence_tokens[target_start:target_end]
                    targets_asp = ['O'] * len(aspects)
                    labels_asp = ['O'] * len(aspects)
                    if len(aspects) == 0:
                        continue
                    aspects[0] = aspects[0].lstrip()
                    sentence_tokens.extend(aspects)
                    target.extend(targets_asp)
                    label.extend(labels_asp)
                                
                elif token_type == "sentence_pair_second":
                    sentence_tokens.append(tokenizer.sep_token)
                    target.append('O')
                    label.append('O')
                    aspects = sentence_tokens[target_start:target_end]
                    targets = target[target_start:target_end]
                    target[target_start:target_end] = ['O'] * len(aspects)
                    labels = ['O'] * len(aspects)
                    if len(aspects) == 0:
                            continue
                    aspects[0] = aspects[0].lstrip()
                    sentence_tokens.extend(aspects)
                    target.extend(targets)
                    label.extend(labels)
                elif token_type == "sentence_pair_first_double_blank":
                    sentence_tokens.append(tokenizer.sep_token)
                    target.append('O')
                    label.append('O')
                    aspects = sentence_tokens[target_start:target_end]
                    if len(aspects) == 0:
                        continue
                    targets = ['O'] * len(aspects)
                    labels = ['O'] * len(aspects)
                    aspects[0] = aspects[0].lstrip()
                    sentence_tokens.extend(aspects)
                    target.extend(targets)
                    label.extend(labels)

                    if bert_type == "roberta":
                        sentence_tokens.insert(target_start,' ')
                        sentence_tokens.insert(target_end + 1, ' ')
                    else:
                        sentence_tokens.insert(target_start,'##')
                        sentence_tokens.insert(target_end + 1, '##')

                    target.insert(target_start,'O')
                    target.insert(target_end + 1, 'O')

                    label.insert(target_start,'O')
                    label.insert(target_end + 1, 'O')

                elif token_type == "sentence_pair_first_double_token":
                    sentence_tokens.append(tokenizer.sep_token)
                    target.append('O')
                    label.append('O')
                    aspects = sentence_tokens[target_start:target_end]
                    if len(aspects) == 0:
                        continue
                    targets = ['O'] * len(aspects)
                    labels = ['O'] * len(aspects)
                    aspects[0] = aspects[0].lstrip()
                    sentence_tokens.extend(aspects)
                    target.extend(targets)
                    label.extend(labels)

                    sentence_tokens.insert(target_start,'<asp>')
                    sentence_tokens.insert(target_end + 1, '</asp>')

                    target.insert(target_start,'O')
                    target.insert(target_end + 1, 'O')

                    label.insert(target_start,'O')
                    label.insert(target_end + 1, 'O')
                elif token_type == "aspect_prompt":
                    for i, prompt_token in enumerate(prompt_sentence_tokens):
                        sentence_tokens.append(prompt_token)
                        target.append('O')
                        label.append('O')
                    aspects = sentence_tokens[target_start:target_end]
                    if len(aspects) == 0:
                        continue
                    targets = ['O'] * len(aspects)
                    labels = ['O'] * len(aspects)
                    # aspects[0] = aspects[0].lstrip()
                    sentence_tokens.extend(aspects)
                    target.extend(targets)
                    label.extend(labels)
                    sentence_tokens.append(' .')
                    target.append('O')
                    label.append('O')
                else:
                    print("No Special Token Need To Add.")
                
                text_tokens_list.append(sentence_tokens)
                target_list.append(target)
                text_target_list.append(sentence_tokens + ['@#$%&##@#'] +target)
                label_list.append(label)
                assert len(label_list) == len(target_list)
                assert len(text_tokens_list) == len(label_list)

                
        if set_type == 'train':
            dev_text_tokens_list = []
            dev_target_list = []
            dev_label_list = []
            
            X_train, X_dev, y_train, y_dev= train_test_split(text_target_list,label_list,test_size=0.2,random_state=0)
            
            text_tokens_list.clear()
            target_list.clear()
            label_list.clear()

            for i, sentence_target in enumerate(X_train):
                special_token_loc = sentence_target.index('@#$%&##@#')
                text_tokens_list.append(sentence_target[:special_token_loc])
                target_list.append(sentence_target[special_token_loc+1:])
                
            for i, sentence_target in enumerate(X_dev):
                special_token_loc = sentence_target.index('@#$%&##@#')
                dev_text_tokens_list.append(sentence_target[:special_token_loc])
                dev_target_list.append(sentence_target[special_token_loc+1:])
            
            label_list = y_train
            dev_label_list = y_dev

            for i in range(len(text_tokens_list)):
                data = tokenize_data_for_bert(text_tokens_list[i], target_list[i], label_list[i], tokenizer, max_length=max_length, token_type=token_type)
                data_dict['train'].append(data)
            
            for i in range(len(dev_text_tokens_list)):
                data = tokenize_data_for_bert(dev_text_tokens_list[i], dev_target_list[i], dev_label_list[i], tokenizer, max_length=max_length, token_type=token_type)
                data_dict['dev'].append(data)
        else:
            for i in range(len(text_tokens_list)):
                data = tokenize_data_for_bert(text_tokens_list[i], target_list[i], label_list[i], tokenizer, max_length=max_length, token_type=token_type)
                data_dict[set_type].append(data)
    
    return data_dict