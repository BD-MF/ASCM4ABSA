# -*- coding: utf-8 -*-
import os
import torch
import shutil
import random
import argparse
import numpy as np
from trainer import Trainer
from models import get_model
from evaluator import Evaluator
from data_utils import build_data_for_bert, build_tokenizer_for_bert, build_embedding_for_bert

def set_dir(args):
    os.makedirs(args.cache_dir, exist_ok=True)
    args.exp_dir = os.path.join(args.output_dir, '{}-{}-{}'.format(args.train_data_name, args.model_name, args.token_type))
    os.makedirs(args.exp_dir, exist_ok=True)
    args.train_data_dir = os.path.join(args.data_dir, args.train_data_name)
    args.test_data_dir = os.path.join(args.data_dir, args.test_data_name)

def set_device(args):
    args.num_gpus = torch.cuda.device_count()
    args.device = torch.device('cuda' if args.num_gpus > 0 else 'cpu')

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

def set_model(args):
    args.model_class, args.input_fields = get_model(args.model_name)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default='train', type=str)
    parser.add_argument('--model_name', default='roberta', type=str)
    parser.add_argument('--train_data_name',
                        default='14lap',
                        type=str,
                        help='14lap, 14res, 15res, 16res')
    parser.add_argument('--test_data_name',
                        default='14lap',
                        type=str,
                        help='14lap, 14res, 15res, 16res')
    parser.add_argument('--learning_rate',
                        default=1e-5,
                        type=float,
                        help='try 5e-5, 2e-5 for BERT or RoBERTa, 1e-3 for others')
    parser.add_argument('--weight_decay', default=1e-5, type=float)  #0.01 #0.00001
    parser.add_argument('--batch_size',
                        default=32,
                        type=int,
                        help='try 16, 32, 64 for BERT models')
    parser.add_argument('--num_repeats',
                        default=5,
                        type=int,
                        help='num of repeated experiments')
    parser.add_argument('--num_train_epochs',
                        default=100,
                        type=int,
                        help='try larger number for non-BERT models')
    parser.add_argument('--num_patience_epochs', default=6, type=int)
    parser.add_argument('--log_interval', default=10, type=int)
    parser.add_argument('--embed_size', default=300, type=int)
    parser.add_argument('--hidden_size', default=300, type=int)
    parser.add_argument('--bert_size', default=768, type=int)
    parser.add_argument('--polarity_size', default=3, type=int)
    parser.add_argument('--seed',
                        default=776,
                        type=int,
                        help='set seed for reproducibility')
    parser.add_argument('--pretrained_model_name_or_path',
                        default='roberta-base',
                        type=str,
                        help='roberta-base, bert-base-uncased')
    parser.add_argument('--max_seq_len', default=128, type=int)
    parser.add_argument('--num_hops', default=3, type=int)
    parser.add_argument('--data_dir', default='datasets', type=str)
    parser.add_argument('--output_dir', default='outputs', type=str)
    parser.add_argument('--cache_dir', default='caches', type=str)
    parser.add_argument('--initializer', default='xavier_uniform_', type=str)
    parser.add_argument('--token_type', default='no', type=str, help='no, before_blank, after_blank, double blank, before_token, after_token, double_token, sentence_pair_first, sentence_pair_second' )
    
    args = parser.parse_args()
    initializers = {
        'xavier_uniform_': torch.nn.init.xavier_uniform_,
        'xavier_normal_': torch.nn.init.xavier_normal_,
        'orthogonal_': torch.nn.init.orthogonal_,
    }

    args.initializer = initializers[args.initializer]
    set_dir(args)
    set_device(args)
    set_seed(args)
    set_model(args)

    if args.mode == 'train':
        bert_type = 'bert'
        if 'roberta' in args.model_name:
            bert_type = 'roberta'
        tokenizer = build_tokenizer_for_bert(args.pretrained_model_name_or_path, cache_dir=args.cache_dir, use_fast=True)
        tokenizer.add_tokens(['<asp>'])
        tokenizer.add_tokens(['</asp>'])
        tokenizer.add_tokens(['##'])
        embedding = build_embedding_for_bert(args.pretrained_model_name_or_path, cache_dir=args.cache_dir)
        embedding.resize_token_embeddings(len(tokenizer))
        data_dict = build_data_for_bert(args.train_data_dir, tokenizer, max_length=args.max_seq_len, token_type=args.token_type, bert_type = bert_type)
        trainer = Trainer(args)
        trainer.run(args, embedding, data_dict['train'], data_dict['dev'])

    elif args.mode == 'evaluate':
        bert_type = 'bert'
        if 'roberta' in args.model_name:
            bert_type = 'roberta'
        tokenizer = build_tokenizer_for_bert(args.pretrained_model_name_or_path, cache_dir=args.cache_dir, use_fast=True)
        tokenizer.add_tokens(['<asp>'])
        tokenizer.add_tokens(['</asp>'])
        tokenizer.add_tokens(['##'])
        embedding = build_embedding_for_bert(args.pretrained_model_name_or_path, cache_dir=args.cache_dir)
        embedding.resize_token_embeddings(len(tokenizer))
        data_dict = build_data_for_bert(args.test_data_dir, tokenizer, max_length=args.max_seq_len, token_type=args.token_type, bert_type = bert_type)
        evaluator = Evaluator(args)
        evaluator.run(args, embedding, data_dict['test'], tokenizer)
