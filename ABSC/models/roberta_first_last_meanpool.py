# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F


class Roberta_First_Last_MeanPool(nn.Module):
    def __init__(self, args, embedding):
        super(Roberta_First_Last_MeanPool, self).__init__()
        self.embed = embedding
        self.ffn = nn.Sequential(
            nn.Linear(args.bert_size, args.bert_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(args.bert_size, args.polarity_size),
        )
        
    
    def forward(self, inputs):
        text_indices, text_mask, aspect_boundary_indices, aspect_indices, aspect_mask = inputs

        txt_embed = self.embed(text_indices, attention_mask=text_mask)[0]
        
        # get hidden states of the first and the last tokens from each aspect
        aspect_hidden_states = torch.gather(txt_embed, 1, aspect_boundary_indices.unsqueeze(2).expand(-1,-1,txt_embed.shape[-1])) #[batch_size,2,emb_size]
        aspect_feat = F.avg_pool1d(aspect_hidden_states.transpose(1, 2), aspect_hidden_states.shape[1]).squeeze()#[batch_size,emb_size]
        out = self.ffn(aspect_feat)

        return out

