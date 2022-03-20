import os
import pickle
import torch
from transformers import XLMRobertaModel
import torch.nn.functional as F

from utils.data_utils import *

import torch.nn as nn

DEVICE = 'cuda:0'

class ImgAttention(nn.Module):
    def __init__(self, hidden_size, drop_out):
        super().__init__()
        self.w_q = nn.Linear(hidden_size, hidden_size)
        self.w_k = nn.Linear(hidden_size, hidden_size)
        self.w_v = nn.Linear(hidden_size, hidden_size)

        self.f_c = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(drop_out)

    def forward(self, img_embedding, mask):
        Q = self.w_q(img_embedding)
        K = self.w_k(img_embedding).permute(0,2,1)
        V = self.w_v(img_embedding)
        max_len = max(mask)
        mask = torch.arange(mask.max().item())[None, :].to(DEVICE) < mask[:, None]
        mask = mask.unsqueeze(dim = 1)
        mask = mask.expand(img_embedding.shape[0], max_len, max_len)
        
        padding_num = torch.ones_like(mask)
        padding_num = -2**31 * padding_num.float()
        
        alpha = torch.matmul(Q, K)

        alpha = torch.where(mask, alpha, padding_num)
        alpha = F.softmax(alpha, dim = 2)
        out = torch.mean(torch.matmul(alpha, V),dim=1)
        return out


class MultiSourceForSememePrediction(nn.Module):
    def __init__(self, model, n_labels, text_hidden_size, img_hidden_size, dropout_p):
        super().__init__()
        self.n_labels = n_labels
        self.text_encoder = XLMRobertaModel.from_pretrained(model)
        self.text_pooler_classification_head = nn.Linear(
            text_hidden_size, n_labels)
        self.text_max_classification_head = nn.Linear(
            text_hidden_size, n_labels)
        self.text_pretrain_classification_head = nn.Linear(
            text_hidden_size, n_labels)
        self.img_classification_head = nn.Linear(img_hidden_size, n_labels)
        self.img_encoder_classification_head = nn.Linear(1000, img_hidden_size)
        self.classification_head = nn.Linear(text_hidden_size+img_hidden_size, n_labels)
        # self.classification_head = nn.Linear(text_hidden_size+img_hidden_size, n_labels, bias=False)
        self.dropout = nn.Dropout(dropout_p)
        self.loss = torch.nn.MultiLabelSoftMarginLoss()

    def forward(self, mode, text_ids, text_mask, img_ids=None, labels=None, mask_idx=None):
        if mode == 'pretrain':
            output = self.text_encoder(
                input_ids=text_ids, attention_mask=text_mask)
            output = output.last_hidden_state
            output = self.dropout(output)
            output = output.gather(1, mask_idx.unsqueeze(1)).squeeze(1)
            output = self.text_pretrain_classification_head(output)
        elif mode == 'train_img':
            output = self.img_classification_head(img_ids)
        elif mode == 'train_text_with_pooler_output':
            output = self.text_encoder(
                input_ids=text_ids, attention_mask=text_mask)
            output = output.pooler_output
            output = self.dropout(output)
            output = self.text_pooler_classification_head(output)
        elif mode == 'train_text_with_last_hidden_state':
            output = self.text_encoder(
                input_ids=text_ids, attention_mask=text_mask)
            output = output.last_hidden_state
            output = self.dropout(output)
            output = self.text_max_classification_head(output)
            mask = text_mask.to(torch.float32).unsqueeze(2)
            output = output * mask + (-1e7) * (1-mask)
            output, _ = torch.max(output, dim=1)
        elif mode == 'train_with_multi_source':
            text_output = self.text_encoder(
                input_ids=text_ids, attention_mask=text_mask)
            text_output = text_output.pooler_output
            output = torch.cat([text_output, img_ids], dim=1)
            output = self.classification_head(output)
        elif mode == 'train_with_multi_source_pro':
            text_output = self.text_encoder(
                input_ids=text_ids, attention_mask=text_mask)
            text_output = text_output.pooler_output
            img_output = self.img_encoder_classification_head(img_ids)
            output = torch.cat([text_output, img_output], dim=1)
            output = self.dropout(output)
            output = self.classification_head(output)
        elif mode == 'training_with_multi_source_add':
            text_output = self.text_encoder(
                input_ids=text_ids, attention_mask=text_mask)
            text_output = text_output.pooler_output
            img_output = self.img_encoder_classification_head(img_ids)
            output = torch.cat([text_output, img_output], dim=1)
            output = self.dropout(output)
            output = self.classification_head(output)
        _, indice = torch.sort(output, descending=True)
        loss = self.loss(output, labels)
        return loss, output, indice


class AttentionMultiSourceForSememePrediction(nn.Module):
    def __init__(self, model, n_labels, text_hidden_size, img_hidden_size, dropout_p):
        super().__init__()
        self.n_labels = n_labels
        self.text_encoder = XLMRobertaModel.from_pretrained(model)
        self.text_pooler_classification_head = nn.Linear(
            text_hidden_size, n_labels)
        self.text_max_classification_head = nn.Linear(
            text_hidden_size, n_labels)
        self.text_pretrain_classification_head = nn.Linear(
            text_hidden_size, n_labels)
        self.img_classification_head = nn.Linear(img_hidden_size, n_labels)
        self.img_encoder_classification_head = nn.Linear(1000, img_hidden_size)
        self.classification_head = nn.Linear(
            text_hidden_size+img_hidden_size, n_labels)
        self.dropout = nn.Dropout(dropout_p)
        self.loss = torch.nn.MultiLabelSoftMarginLoss()
        self.img_attention = ImgAttention(1000, dropout_p)

    def forward(self, mode, text_ids, text_mask, img_ids=None, img_mask=None, labels=None, mask_idx=None):
        if mode == 'pretrain':
            output = self.text_encoder(
                input_ids=text_ids, attention_mask=text_mask)
            output = output.last_hidden_state
            output = self.dropout(output)
            output = output.gather(1, mask_idx.unsqueeze(1)).squeeze(1)
            output = self.text_pretrain_classification_head(output)
        elif mode == 'attention_multi_source':
            text_output = self.text_encoder(
                input_ids=text_ids, attention_mask=text_mask)
            text_output = text_output.pooler_output
            img_output = self.img_attention(img_ids, img_mask)
            img_output = self.img_encoder_classification_head(img_output)
            output = torch.cat([text_output, img_output], dim=1)
            output = self.dropout(output)
            output = self.classification_head(output)
        _, indice = torch.sort(output, descending=True)
        loss = self.loss(output, labels)
        return loss, output, indice
