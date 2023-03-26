#!/bin/bash

for token_type in  aspect_prompt double_token sentence_pair_first
    do  
        for dn in 14lap 14res
            do  
                CUDA_VISIBLE_DEVICES=2 python run.py --model_name towe_first_last_meanpool_roberta --train_data_name ${dn} --mode train --learning_rate 5e-5 --weight_decay 0 --batch_size 64 --token_type ${token_type} --num_patience_epochs 5
                CUDA_VISIBLE_DEVICES=2 python run.py --model_name towe_first_last_meanpool_roberta --train_data_name ${dn} --test_data_name ${dn} --mode evaluate --learning_rate 5e-5 --weight_decay 0 --batch_size 64 --token_type ${token_type}   --num_patience_epochs 5 
                CUDA_VISIBLE_DEVICES=2 python run.py --model_name towe_first_last_meanpool_roberta --train_data_name ${dn} --test_data_name aoptrts-${dn} --mode evaluate --learning_rate 5e-5 --weight_decay 0 --batch_size 64 --token_type ${token_type}   --num_patience_epochs 5 

                CUDA_VISIBLE_DEVICES=2 python run.py --model_name towe_first_last_meanpool_bert --train_data_name ${dn} --mode train --learning_rate 5e-5 --weight_decay 0 --batch_size 64 --token_type ${token_type} --pretrained_model_name_or_path  bert-base-uncased --num_patience_epochs 5
                CUDA_VISIBLE_DEVICES=2 python run.py --model_name towe_first_last_meanpool_bert --train_data_name ${dn} --test_data_name ${dn} --mode evaluate --learning_rate 5e-5 --weight_decay 0 --batch_size 64 --token_type ${token_type} --pretrained_model_name_or_path  bert-base-uncased  --num_patience_epochs 5
                CUDA_VISIBLE_DEVICES=2 python run.py --model_name towe_first_last_meanpool_bert --train_data_name ${dn} --test_data_name aoptrts-${dn} --mode evaluate --learning_rate 5e-5 --weight_decay 0 --batch_size 64 --token_type ${token_type} --pretrained_model_name_or_path  bert-base-uncased  --num_patience_epochs 5
            done 
    done


