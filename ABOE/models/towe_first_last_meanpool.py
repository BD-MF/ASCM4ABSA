# -*- coding: utf-8 -*-

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class TOWE_First_Last_MeanPool(nn.Module):
    def __init__(self, args, embedding):
        super(TOWE_First_Last_MeanPool, self).__init__()
        self.embed = embedding
        
        self.ffn = nn.Sequential(
            nn.Linear(2*args.bert_size, args.bert_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(args.bert_size, args.polarity_size),
        )

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


    def forward(self, inputs):
        text_tokens, text_tokens_mask, target_tag_list, target_tag_mask, label_tag_list,  label_tag_mask, aspect_boundary_indices = inputs
        
        txt_embed = self.embed(text_tokens, attention_mask=text_tokens_mask, output_attentions=True)[0]#[batch_size, max_seq_len, bert_size]
        
        aspect_hidden_states = torch.gather(txt_embed, 1, aspect_boundary_indices.unsqueeze(2).expand(-1,-1,txt_embed.shape[-1])) #[batch_size,2,bert_size]
        aspect_feat = F.avg_pool1d(aspect_hidden_states.transpose(1, 2), aspect_hidden_states.shape[1]).squeeze() #[batch_size,bert_size]
        aspect_feat = aspect_feat.unsqueeze(1)#[batch_size,1,bert_size]
        aspect_feat = aspect_feat.repeat(1, txt_embed.shape[1] ,1)#[batch_size,max_seq_len,bert_size]
        
        text_aspect_embed = torch.cat((txt_embed,aspect_feat), -1)#[batch_size,max_seq_len,bert_size*2]
        
        out = self.ffn(text_aspect_embed)
        return out