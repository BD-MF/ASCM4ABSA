# -*- coding: utf-8 -*-

import os
import json
import nltk
import spacy
import random
import pickle
import numpy as np
from logger import Logger
from spacy.tokens import Doc
from transformers import AutoTokenizer, AutoModel, AutoConfig


def build_embedding(data_dir, token2idx, embed_size):
    if os.path.exists(os.path.join(data_dir, 'embedding.pt')):
        print('>> loading embedding: {}'.format(
            os.path.join(data_dir, 'embedding.pt')))
        embedding = pickle.load(
            open(os.path.join(data_dir, 'embedding.pt'), 'rb'))
    else:
        # words not found in embedding index will be randomly initialized.
        embedding = np.random.uniform(-1 / np.sqrt(embed_size),
                                      1 / np.sqrt(embed_size),
                                      (len(token2idx), embed_size))
        embedding[0, :] = np.zeros((1, embed_size))
        fn = 'glove.840B.300d.txt'
        print('>> loading word vectors')
        word2vec = {}
        with open(fn, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                numbers = line.rstrip().split()
                token, vec = ' '.join(numbers[:-embed_size]), numbers[-embed_size:]
                if token in token2idx.keys():
                    word2vec[token] = np.asarray(vec, dtype=np.float32)
        print('>> building embedding: {}'.format(
            os.path.join(data_dir, 'embedding.pt')))
        for token, i in token2idx.items():
            vec = word2vec.get(token)
            if vec is not None:
                embedding[i] = vec
        pickle.dump(embedding,open(os.path.join(data_dir, 'embedding.pt'), 'wb'))
    return embedding

def build_embedding_for_bert(data_dir, cache_dir='cahces'):
    config = AutoConfig.from_pretrained(data_dir, cache_dir=cache_dir)                           
    embedding = AutoModel.from_pretrained(data_dir, config=config, cache_dir=cache_dir)
    return embedding

class Tokenizer(object):
    def __init__(self, token2idx=None):
        if token2idx is None:
            self.token2idx = {}
            self.idx2token = {}
            self.idx = 0
            self.token2idx['<pad>'] = self.idx
            self.idx2token[self.idx] = '<pad>'
            self.idx += 1
            self.token2idx['<unk>'] = self.idx
            self.idx2token[self.idx] = '<unk>'
            self.idx += 1
        else:
            self.token2idx = token2idx
            self.idx2token = {v: k for k, v in token2idx.items()}

    def fit_on_text(self, text):
        tokens = text.split()
        for token in tokens:
            if token not in self.token2idx:
                self.token2idx[token] = self.idx
                self.idx2token[self.idx] = token
                self.idx += 1

    def convert_tokens_to_ids(self, tokens):
        return [self.token2idx[t] if t in self.token2idx else 1 for t in tokens]
    
    @staticmethod
    def tokenize(text):
        return text.lower().split()

    def tokenize_sentence(self,text):
        return nltk.word_tokenize(text.lower(), preserve_line=True)

    @staticmethod
    def nltk_word_tokenize(text):
        return nltk.word_tokenize(text.lower(), preserve_line=True)

    def __call__(self, text):
        return self.convert_tokens_to_ids(self.tokenize(text))

def build_tokenizer(data_dir):
    if os.path.exists(os.path.join(data_dir, 'token2idx.pt')):
        print('>> loading {} tokenizer'.format(data_dir))
        with open(os.path.join(data_dir, 'token2idx.pt'), 'rb') as f:
            token2idx = pickle.load(f)
            tokenizer = Tokenizer(token2idx=token2idx)
    else:
        all_text = ''
        set_types = ['train', 'dev', 'test']
        for set_type in set_types:
            with open(os.path.join(data_dir, '{}.json'.format(set_type)),'r',encoding='utf-8') as f:
                set_dict = json.load(f)
                for k in set_dict:
                    term = set_dict[k]['term'].lower()
                    asp_from = set_dict[k]['sentence'].lower().index(term)
                    asp_to = asp_from + len(term)
                    text_left = set_dict[k]['sentence'][:asp_from]
                    text_right = set_dict[k]['sentence'][asp_to:]
                    aspect = set_dict[k]['sentence'][asp_from:asp_to]
                    text_left = " ".join(Tokenizer.nltk_word_tokenize(text_left)).lower()
                    text_right = " ".join(Tokenizer.nltk_word_tokenize(text_right)).lower()
                    aspect = " ".join(Tokenizer.nltk_word_tokenize(aspect)).lower()
                    if len(text_left) != 0:
                        text_left += " "
                    if len(text_right) != 0:
                        text_right = " " + text_right 
                    final_text = text_left + aspect + text_right
                    all_text += (final_text + " ")
        tokenizer = Tokenizer()
        tokenizer.fit_on_text(all_text)
        print('>> building {} tokenizer'.format(data_dir))
        with open(os.path.join(data_dir, 'token2idx.pt'), 'wb') as f:
            pickle.dump(tokenizer.token2idx, f)
    return tokenizer

def build_tokenizer_for_bert(data_dir, cache_dir='caches', use_fast=True):
    tokenizer = AutoTokenizer.from_pretrained(data_dir, cache_dir=cache_dir, use_fast=True)
    return tokenizer

def truncate_and_pad(indices, max_length=128, pad_idx=0):
    # if len(indices) > max_length:
    #     indices = indices[:max_length-1] + [indices[-1]]
    if len(indices) > max_length:
        sep_idx = indices[-1]
        sep_pos = indices.index(sep_idx) # first sep pos
        truncate_len = len(indices) - max_length
        indices = indices[:sep_pos - truncate_len] + indices[sep_pos:]
    _len = len(indices)
    indices = indices + [pad_idx] * (max_length - _len)
    mask = [1] * _len + [0] * (max_length - _len)
    return indices, mask

class WhitespaceTokenizer(object):
    def __init__(self, vocab):
        self.vocab = vocab

    def __call__(self, text):
        words = text.split()
        # All tokens 'own' a subsequent space character in this tokenizer
        spaces = [True] * len(words)
        return Doc(self.vocab, words=words, spaces=spaces)

nlp = spacy.load('en_core_web_sm')
nlp.tokenizer = WhitespaceTokenizer(nlp.vocab)

def dependency_adj_matrix(text):
    # https://spacy.io/docs/usage/processing-text
    tokens = nlp(text)
    words = text.split()
    matrix = np.zeros((len(words), len(words))).astype('float32')
    assert len(words) == len(list(tokens))
    
    for token in tokens:
        matrix[token.i][token.i] = 1
        for child in token.children:
            matrix[token.i][child.i] = 1
    return matrix

def dependency_graph_adj_matrix(text):
    # https://spacy.io/docs/usage/processing-text
    tokens = nlp(text)
    words = text.split()
    matrix = np.zeros((len(words), len(words))).astype('float32')
    assert len(words) == len(list(tokens))

    for token in tokens:
        matrix[token.i][token.i] = 1
        for child in token.children:
            matrix[token.i][child.i] = 1
            matrix[child.i][token.i] = 1
    return matrix

def build_data(data_dir, tokenizer, max_length=128):
    data_dict = {'train': [], 'dev': [], 'test': []}
    set_types = ['train', 'dev', 'test']
    polarity_map = {'negative': 0, 'neutral': 1, 'positive': 2}
    
    for set_type in set_types:
        fname = os.path.join(data_dir, '{}.json'.format(set_type))
        with open(fname, 'r', encoding='utf-8') as f:
            set_dict = json.load(f)
            for i, k in enumerate(set_dict):
                sentence = set_dict[k]['sentence']
                text_left = sentence[:set_dict[k]['from']]
                text_right = sentence[set_dict[k]['to']:]
                aspect = sentence[set_dict[k]['from']:set_dict[k]['to']]

                if aspect.lower() != set_dict[k]['term'].lower():
                    if set_dict[k]['term'].lower() not in sentence.lower():
                        continue
                    else:
                        text_left = sentence[:sentence.lower().index(set_dict[k]['term'].lower())].lower()
                        text_right = sentence[sentence.lower().index(set_dict[k]['term'].lower())+len(set_dict[k]['term']):].lower()
                        aspect = set_dict[k]['term'].lower()

                text_left = " ".join(nltk.word_tokenize(text_left, preserve_line=True))
                text_right = " ".join(nltk.word_tokenize(text_right, preserve_line=True))
                aspect = " ".join(nltk.word_tokenize(aspect, preserve_line=True))
                if len(text_left) != 0:
                    text_left += " "
                if len(text_right) != 0:
                    text_right = " " + text_right
                final_text = text_left + aspect + text_right
                
                #dependency_tree = dependency_adj_matrix(final_text)
                #dependency_graph = dependency_graph_adj_matrix(final_text)
                #assert len(dependency_tree) == len(final_text.split())

                left_indices = tokenizer(text_left)
                right_indices = tokenizer(text_right)
                aspect_indices = tokenizer(aspect)

                sentence_indices = tokenizer(final_text)
                text_len = len(sentence_indices)
                assert len(left_indices + right_indices + aspect_indices) == text_len
                #assert  len(dependency_tree) == text_len

                text_indices, text_mask = truncate_and_pad(left_indices + aspect_indices + right_indices, max_length=max_length)
                #aspect_position_mask, _ = truncate_and_pad([0] * len(left_indices) + [1] * len(aspect_indices) + [0] * len(right_indices), max_length=max_length)
                aspect_boundary_indices = [len(left_indices), len(left_indices) + len(aspect_indices) - 1]
                aspect_indices, aspect_mask = truncate_and_pad(aspect_indices, max_length=max_length)
                left_indices, _ = truncate_and_pad(left_indices, max_length=max_length)

                polarity = polarity_map[set_dict[k]['polarity']]
                dependency_tree = [[]]
                dependency_graph = [[]]

                data = {
                    'text_indices': text_indices,
                    'text_mask': text_mask,
                    'text_len':text_len,
                    'aspect_boundary_indices': aspect_boundary_indices,
                    'aspect_indices': aspect_indices,
                    'aspect_mask': aspect_mask,
                    'left_indices':left_indices,
                    'polarity': polarity,
                    'dependency_tree': dependency_tree,
                    'dependency_graph':dependency_graph,
                }

                data_dict[set_type].append(data)
    return data_dict


def build_data_for_bert(data_dir, tokenizer, max_length=128,token_type="before_blank", bert_type = "roberta"):
    data_dict = {'train': [], 'dev': [], 'test': []}
    set_types = ['train', 'dev', 'test']
    polarity_map = {'negative': 0, 'neutral': 1, 'positive': 2}

    for set_type in set_types:
        fname = os.path.join(data_dir, '{}.json'.format(set_type))
        with open(fname, 'r', encoding='utf-8') as f:
            set_dict = json.load(f)
            for k in set_dict:
                sentence = set_dict[k]['sentence']
                text_left = sentence[:set_dict[k]['from']]
                text_right = sentence[set_dict[k]['to']:]
                aspect = sentence[set_dict[k]['from']:set_dict[k]['to']]
                if aspect.lower() != set_dict[k]['term'].lower():
                    if set_dict[k]['term'].lower() not in sentence.lower():
                        continue
                    else:
                        text_left = sentence[:sentence.lower().index(set_dict[k]['term'].lower())].lower()
                        text_right = sentence[sentence.lower().index(set_dict[k]['term'].lower())+len(set_dict[k]['term']):].lower()
                        aspect = set_dict[k]['term'].lower()
                text_left_ = " ".join(nltk.word_tokenize(text_left, preserve_line=True))
                text_right_ = " ".join(nltk.word_tokenize(text_right, preserve_line=True))
                aspect_ = " ".join(nltk.word_tokenize(aspect, preserve_line=True))
                if len(text_left_) != 0:
                    text_left_ += " "
                if len(text_right_) != 0:
                    text_right_ = " " + text_right_ 
                final_text =  text_left_ + aspect_ + text_right_
                concat_bert_indices = []
                concat_segments_indices = []
                
                prompt_sentence = "The target aspect is "
                prompt_sentence_tokens = prompt_sentence.strip().split()
                for i, token in enumerate(prompt_sentence_tokens):
                    prompt_sentence_tokens[i] =  ' ' + token

                if "sentence_pair" not in token_type:
                    if bert_type == "roberta":
                        left_tokens = []
                        for i,left_token in enumerate(text_left_.split()):
                            if i != 0:
                                left_token = ' '+left_token
                            left_tokens.append(left_token)
                        if token_type == "before_blank" or token_type == "double_blank":
                            left_tokens.append(' ')
                        elif token_type == "before_token" or token_type == "double_token":
                            left_tokens.append('<asp>')
                        aspect_tokens = []
                        for i, aspect_token in enumerate(aspect_.split()):
                            # if i != 0:
                            aspect_token = ' ' + aspect_token
                            aspect_tokens.append(aspect_token)
                        right_tokens = []
                        if token_type == "after_blank" or token_type == "double_blank":
                            right_tokens.append(' ')
                        elif token_type == "after_token" or token_type == "double_token":
                            right_tokens.append('</asp>')
                        for i,right_token in enumerate(text_right_.split()):
                            right_token = ' ' + right_token
                            right_tokens.append(right_token)
                        
                        if token_type == "aspect_prompt":
                            right_tokens.extend(prompt_sentence_tokens)
                            right_tokens.extend(aspect_tokens)
                            right_tokens.append(' .')

                        blank_split_sentence = left_tokens + aspect_tokens + right_tokens
                        token_map = {}
                        text_tokens = []
                        for j, token in enumerate(blank_split_sentence): 
                            token_pieces = tokenizer.tokenize(token)
                            token_map[j] = (len(text_tokens),len(text_tokens) + len(token_pieces) - 1)
                            text_tokens.extend(token_pieces)
                        
                        text_indices = tokenizer.convert_tokens_to_ids(text_tokens)
                        text_len = len(text_indices) + 2
                        text_indices, text_mask = truncate_and_pad([tokenizer.cls_token_id] +text_indices + [tokenizer.sep_token_id], max_length=max_length, pad_idx=tokenizer.pad_token_id)
                        
                        left_text_tokens = []
                        for j ,token in enumerate(left_tokens): 
                            token_pieces = tokenizer.tokenize(token)
                            left_text_tokens.extend(token_pieces)
                        
                        aspect_text_tokens = []
                        for j ,token in enumerate(aspect_tokens): 
                            token_pieces = tokenizer.tokenize(token)
                            aspect_text_tokens.extend(token_pieces)

                        aspect_boundary_indices = [len(left_text_tokens) + 1, len(left_text_tokens) + len(aspect_text_tokens)]
                        aspect_indices = tokenizer.convert_tokens_to_ids(aspect_text_tokens)
                        aspect_indices, aspect_mask = truncate_and_pad([tokenizer.cls_token_id] + aspect_indices  + [tokenizer.sep_token_id], max_length=max_length, pad_idx=tokenizer.pad_token_id)
                    else:#TODO berta_type = bert
                        left_tokens = text_left_.split()
                        if token_type == "before_blank" or token_type == "double_blank":
                            left_tokens.append('##')
                        elif token_type == "before_token" or token_type == "double_token":
                            left_tokens.append('<asp>')
                        aspect_tokens = aspect_.split()
                        right_tokens = text_right_.split()
                        if token_type == "after_blank" or token_type == "double_blank":
                            right_tokens.insert(0, '##')
                        elif token_type == "after_token" or token_type == "double_token":
                            right_tokens.insert(0, '</asp>')
                        
                        if token_type == "aspect_prompt":
                            right_tokens.extend(prompt_sentence_tokens)
                            right_tokens.extend(aspect_tokens)
                            right_tokens.append(' .')
                        
                        blank_split_sentence = left_tokens + aspect_tokens + right_tokens
                        token_map = {}
                        text_tokens = []
                        for j, token in enumerate(blank_split_sentence): 
                            token_pieces = tokenizer.tokenize(token)
                            token_map[j] = (len(text_tokens),len(text_tokens) + len(token_pieces) - 1)
                            text_tokens.extend(token_pieces)
                        text_indices = tokenizer.convert_tokens_to_ids(text_tokens)
                        text_len = len(text_indices) + 2
                        text_indices, text_mask = truncate_and_pad([tokenizer.cls_token_id] +text_indices + [tokenizer.sep_token_id], max_length=max_length, pad_idx=tokenizer.pad_token_id)
                        left_text_tokens = []
                        for j ,token in enumerate(left_tokens): 
                            token_pieces = tokenizer.tokenize(token)
                            left_text_tokens.extend(token_pieces)
                        aspect_text_tokens = []
                        for j ,token in enumerate(aspect_tokens): 
                            token_pieces = tokenizer.tokenize(token)
                            aspect_text_tokens.extend(token_pieces)
                        aspect_boundary_indices = [len(left_text_tokens) + 1, len(left_text_tokens) + len(aspect_text_tokens)]
                        aspect_indices = tokenizer.convert_tokens_to_ids(aspect_text_tokens)
                        aspect_indices, aspect_mask = truncate_and_pad([tokenizer.cls_token_id] + aspect_indices  + [tokenizer.sep_token_id], max_length=max_length, pad_idx=tokenizer.pad_token_id)
                
                        #TODO  concat_bert_indices & concat_segments_indices
                        first_aspect_tokens = aspect_.split()
                        right_tokens.append(tokenizer.sep_token)
                        second_aspect_tokens = aspect_.split() 
                        concat_bert_tokens = left_tokens + first_aspect_tokens +  right_tokens + second_aspect_tokens
                            
                        concat_token_map = {}
                        concat_text_tokens = []
                        for j, token in enumerate(concat_bert_tokens): 
                            token_pieces = tokenizer.tokenize(token)
                            concat_token_map[j] = (len(concat_text_tokens),len(concat_text_tokens) + len(token_pieces) - 1)
                            concat_text_tokens.extend(token_pieces)
                        concat_bert_indices = tokenizer.convert_tokens_to_ids(concat_text_tokens)
                        concat_bert_indices, concat_bert_mask = truncate_and_pad([tokenizer.cls_token_id] +concat_bert_indices + [tokenizer.sep_token_id], max_length=max_length, pad_idx=tokenizer.pad_token_id)
                        
                        concat_segments_indices = [0] * text_len + [1] * (len(aspect_text_tokens) + 1)
                        concat_segments_indices, _ = truncate_and_pad(concat_segments_indices, max_length=max_length, pad_idx=tokenizer.pad_token_id)
                        
                else: #TODO token_type in sentence_pair
                    if token_type == "sentence_pair_first":#取第一个aspect
                        if bert_type == "roberta":
                            left_tokens = []
                            for i,left_token in enumerate(text_left_.split()):
                                if i != 0:
                                    left_token = ' '+left_token
                                left_tokens.append(left_token)
                            first_aspect_tokens = []
                            for i, aspect_token in enumerate(aspect_.split()):
                                aspect_token = ' '+ aspect_token
                                first_aspect_tokens.append(aspect_token)
                            right_tokens = []
                            for i,right_token in enumerate(text_right_.split()):
                                right_token = ' ' + right_token
                                right_tokens.append(right_token)
                            right_tokens.append(tokenizer.sep_token)
                            second_aspect_tokens = []
                            for i, aspect_token in enumerate(aspect_.split()):
                                if i != 0:
                                    aspect_token = ' ' + aspect_token
                                second_aspect_tokens.append(aspect_token)
                                
                            blank_split_sentence = left_tokens + first_aspect_tokens + right_tokens + second_aspect_tokens
                            
                            token_map = {}
                            text_tokens = []
                            for j, token in enumerate(blank_split_sentence): 
                                token_pieces = tokenizer.tokenize(token)
                                token_map[j] = (len(text_tokens),len(text_tokens) + len(token_pieces) - 1)
                                text_tokens.extend(token_pieces)
                            
                            text_indices = tokenizer.convert_tokens_to_ids(text_tokens)
                            text_len = len(text_indices) + 2
                            text_indices, text_mask = truncate_and_pad([tokenizer.cls_token_id] +text_indices + [tokenizer.sep_token_id], max_length=max_length, pad_idx=tokenizer.pad_token_id)
                            
                            left_text_tokens = []
                            for j ,token in enumerate(left_tokens): 
                                token_pieces = tokenizer.tokenize(token)
                                left_text_tokens.extend(token_pieces)
                            aspect_text_tokens = []
                            for j ,token in enumerate(first_aspect_tokens): 
                                token_pieces = tokenizer.tokenize(token)
                                aspect_text_tokens.extend(token_pieces)

                            aspect_boundary_indices = [len(left_text_tokens) + 1, len(left_text_tokens) + len(aspect_text_tokens)]
                            aspect_indices = tokenizer.convert_tokens_to_ids(aspect_text_tokens)
                            aspect_indices, aspect_mask = truncate_and_pad([tokenizer.cls_token_id] + aspect_indices  + [tokenizer.sep_token_id], max_length=max_length, pad_idx=tokenizer.pad_token_id)
                        else:#TODO berta_type=bert
                            left_tokens = text_left_.split()
                            first_aspect_tokens = aspect_.split()
                            right_tokens = text_right_.split()
                            right_tokens.append(tokenizer.sep_token)
                            second_aspect_tokens = aspect_.split()
                            
                            blank_split_sentence = left_tokens + first_aspect_tokens + right_tokens + second_aspect_tokens
                            
                            token_map = {}
                            text_tokens = []
                            for j, token in enumerate(blank_split_sentence): 
                                token_pieces = tokenizer.tokenize(token)
                                token_map[j] = (len(text_tokens),len(text_tokens) + len(token_pieces) - 1)
                                text_tokens.extend(token_pieces)
                            
                            text_indices = tokenizer.convert_tokens_to_ids(text_tokens)
                            text_len = len(text_indices) + 2
                            text_indices, text_mask = truncate_and_pad([tokenizer.cls_token_id] +text_indices + [tokenizer.sep_token_id], max_length=max_length, pad_idx=tokenizer.pad_token_id)
                            
                            left_text_tokens = []
                            for j ,token in enumerate(left_tokens): 
                                token_pieces = tokenizer.tokenize(token)
                                left_text_tokens.extend(token_pieces)
                            aspect_text_tokens = []
                            for j ,token in enumerate(first_aspect_tokens): 
                                token_pieces = tokenizer.tokenize(token)
                                aspect_text_tokens.extend(token_pieces)

                            aspect_boundary_indices = [len(left_text_tokens) + 1, len(left_text_tokens) + len(aspect_text_tokens)]
                            aspect_indices = tokenizer.convert_tokens_to_ids(aspect_text_tokens)
                            aspect_indices, aspect_mask = truncate_and_pad([tokenizer.cls_token_id] + aspect_indices  + [tokenizer.sep_token_id], max_length=max_length, pad_idx=tokenizer.pad_token_id)
                    else:#取第二个aspect <s>sentence</s>aspect</s>
                        if bert_type == "roberta":
                            left_tokens = []
                            for i,left_token in enumerate(text_left_.split()):
                                if i != 0:
                                    left_token = ' '+left_token
                                left_tokens.append(left_token)
                            first_aspect_tokens = []
                            for i, aspect_token in enumerate(aspect_.split()):
                                aspect_token = ' '+ aspect_token
                                first_aspect_tokens.append(aspect_token)
                            right_tokens = []
                            for i,right_token in enumerate(text_right_.split()):
                                right_token = ' ' + right_token
                                right_tokens.append(right_token)
                            right_tokens.append(tokenizer.sep_token)
                            second_aspect_tokens = []
                            for i, aspect_token in enumerate(aspect_.split()):
                                if i != 0:
                                    aspect_token = ' ' + aspect_token
                                second_aspect_tokens.append(aspect_token)
                            
                            blank_split_sentence = left_tokens + first_aspect_tokens +  right_tokens + second_aspect_tokens
                            
                            token_map = {}
                            text_tokens = []
                            for j, token in enumerate(blank_split_sentence): 
                                token_pieces = tokenizer.tokenize(token)
                                token_map[j] = (len(text_tokens),len(text_tokens) + len(token_pieces) - 1)
                                text_tokens.extend(token_pieces)
                            
                            text_indices = tokenizer.convert_tokens_to_ids(text_tokens)
                            text_len = len(text_indices) + 2
                            text_indices, text_mask = truncate_and_pad([tokenizer.cls_token_id] +text_indices + [tokenizer.sep_token_id], max_length=max_length, pad_idx=tokenizer.pad_token_id)
                            
                            left_text_tokens = []
                            for j ,token in enumerate(left_tokens): 
                                token_pieces = tokenizer.tokenize(token)
                                left_text_tokens.extend(token_pieces)
                            first_aspect_text_tokens = []
                            for j ,token in enumerate(first_aspect_tokens): 
                                token_pieces = tokenizer.tokenize(token)
                                first_aspect_text_tokens.extend(token_pieces)
                            right_text_tokens = []
                            for j, token in enumerate(right_tokens):
                                token_pieces = tokenizer.tokenize(token)
                                right_text_tokens.extend(token_pieces)
                            second_aspect_text_tokens = []
                            for j ,token in enumerate(second_aspect_tokens): 
                                token_pieces = tokenizer.tokenize(token)
                                second_aspect_text_tokens.extend(token_pieces)
                        
                            aspect_boundary_indices = [len(left_text_tokens + first_aspect_text_tokens +  right_text_tokens) + 1, len(left_text_tokens + first_aspect_text_tokens +  right_text_tokens) + len(second_aspect_text_tokens)]
                            aspect_indices = tokenizer.convert_tokens_to_ids(second_aspect_text_tokens)
                            aspect_indices, aspect_mask = truncate_and_pad([tokenizer.cls_token_id] + aspect_indices  + [tokenizer.sep_token_id], max_length=max_length, pad_idx=tokenizer.pad_token_id)
                        else:#TODO berta_type=bert
                            left_tokens = text_left_.split()
                            first_aspect_tokens = aspect_.split()
                            right_tokens = text_right_.split()
                            right_tokens.append(tokenizer.sep_token)
                            second_aspect_tokens = aspect_.split()
                            
                            blank_split_sentence = left_tokens + first_aspect_tokens +  right_tokens + second_aspect_tokens
                            
                            token_map = {}
                            text_tokens = []
                            for j, token in enumerate(blank_split_sentence): 
                                token_pieces = tokenizer.tokenize(token)
                                token_map[j] = (len(text_tokens),len(text_tokens) + len(token_pieces) - 1)
                                text_tokens.extend(token_pieces)
                            
                            text_indices = tokenizer.convert_tokens_to_ids(text_tokens)
                            text_len = len(text_indices) + 2
                            text_indices, text_mask = truncate_and_pad([tokenizer.cls_token_id] +text_indices + [tokenizer.sep_token_id], max_length=max_length, pad_idx=tokenizer.pad_token_id)
                            
                            left_text_tokens = []
                            for j ,token in enumerate(left_tokens): 
                                token_pieces = tokenizer.tokenize(token)
                                left_text_tokens.extend(token_pieces)
                            first_aspect_text_tokens = []
                            for j ,token in enumerate(first_aspect_tokens): 
                                token_pieces = tokenizer.tokenize(token)
                                first_aspect_text_tokens.extend(token_pieces)
                            right_text_tokens = []
                            for j, token in enumerate(right_tokens):
                                token_pieces = tokenizer.tokenize(token)
                                right_text_tokens.extend(token_pieces)
                            second_aspect_text_tokens = []
                            for j ,token in enumerate(second_aspect_tokens): 
                                token_pieces = tokenizer.tokenize(token)
                                second_aspect_text_tokens.extend(token_pieces)
                            aspect_boundary_indices = [len(left_text_tokens + first_aspect_text_tokens +  right_text_tokens) + 1, len(left_text_tokens + first_aspect_text_tokens +  right_text_tokens) + len(second_aspect_text_tokens)]
                            aspect_indices = tokenizer.convert_tokens_to_ids(second_aspect_text_tokens)
                            aspect_indices, aspect_mask = truncate_and_pad([tokenizer.cls_token_id] + aspect_indices  + [tokenizer.sep_token_id], max_length=max_length, pad_idx=tokenizer.pad_token_id)
                polarity = polarity_map[set_dict[k]['polarity']]
                
                new_dependency_tree = [[]]
                new_dependency_graph = [[]]
                #TODO
                text_len = 128
                data = {
                    'text_indices': text_indices,
                    'text_mask': text_mask,
                    'text_len':text_len,
                    #'aspect_position_mask': aspect_position_mask,
                    'aspect_boundary_indices': aspect_boundary_indices,
                    'aspect_indices': aspect_indices,
                    'aspect_mask': aspect_mask,
                    'polarity': polarity,
                    'dependency_tree': new_dependency_tree,
                    'dependency_graph': new_dependency_graph,
                    'concat_bert_indices':concat_bert_indices,
                    'concat_segments_indices':concat_segments_indices,
                }
                data_dict[set_type].append(data)
    
    return data_dict