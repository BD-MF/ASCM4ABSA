#!/bin/bash

for dn in laptop rest
    do
        for token_type in double_token sentence_pair_first aspect_prompt
            do
                CUDA_VISIBLE_DEVICES=2 python run.py --model_name bert_first_last_meanpool --train_data_name ${dn} --test_data_name ${dn} --mode train --learning_rate 1e-5 --weight_decay 0.0 --batch_size 64 --token_type ${token_type} --pretrained_model_name_or_path bert-base-uncased
                CUDA_VISIBLE_DEVICES=2 python run.py --model_name bert_first_last_meanpool --train_data_name ${dn} --test_data_name ${dn} --mode evaluate --learning_rate 1e-5 --weight_decay 0.0 --batch_size 64 --token_type ${token_type} --pretrained_model_name_or_path bert-base-uncased
                CUDA_VISIBLE_DEVICES=2 python run.py --model_name bert_first_last_meanpool --train_data_name ${dn} --test_data_name arts_${dn} --mode evaluate --learning_rate 1e-5 --weight_decay 0.0 --batch_size 64 --token_type ${token_type} --pretrained_model_name_or_path bert-base-uncased
                      
                CUDA_VISIBLE_DEVICES=2 python run.py --model_name roberta_first_last_meanpool --train_data_name ${dn} --test_data_name ${dn} --mode train --learning_rate 1e-5 --weight_decay 0.0 --batch_size 64 --token_type ${token_type}
                CUDA_VISIBLE_DEVICES=2 python run.py --model_name roberta_first_last_meanpool --train_data_name ${dn} --test_data_name arts_${dn} --mode evaluate --learning_rate 1e-5 --weight_decay 0.0 --batch_size 64 --token_type ${token_type}
                CUDA_VISIBLE_DEVICES=2 python run.py --model_name roberta_first_last_meanpool --train_data_name ${dn} --test_data_name ${dn} --mode evaluate --learning_rate 1e-5 --weight_decay 0.0 --batch_size 64 --token_type ${token_type}
            done 
    done
