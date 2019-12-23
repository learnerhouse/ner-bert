from __future__ import print_function
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import *


class LSTM_attention(nn.Module):
    '''
    args:
        lstm_hidden_dim     : lstm 的隐层大小
        num_heads           : the head number of multihead 
        hidden_dim          : attention_dim
        
        lstm_out            : 前级输出
        label_embs          : label matrix size = [seq_length,batch_size,embed_size]
        word_seq_lengths    : length after padding
        lstm_hidden         : lstm 隐层 (num_layers * num_directions, batch, hidden_size) * 2
    
    return:
        forward:
        lstm_out            : size = [batch_size,seq_length,2*embed_size]
    '''


    def __init__(self,embed_size,hidden_dim,num_heads,att_hidden_dim,dropout_rate,mode):
        super(LSTM_attention, self).__init__()
        self.mode = mode
        self.lstm = nn.LSTM(embed_size,hidden_dim // 4, num_layers=1, batch_first=True, bidirectional=True)
        #self.slf_attn = multihead_attention(data.HP_hidden_dim,num_heads = data.num_attention_head, dropout_rate=data.HP_dropout)
        self.label_attn = multihead_attention(att_hidden_dim, num_heads=num_heads,dropout_rate=dropout_rate)
        self.droplstm = nn.Dropout(dropout_rate)
        self.lstm =self.lstm.cuda()
        self.label_attn = self.label_attn.cuda()


    def forward(self,input_,label_embs,input_mask,lstm_hidden=None):
        length = input_mask.sum(-1)
        sorted_lengths, sorted_idx = torch.sort(length, descending=True)
        input_ = input_[sorted_idx]
        _, reversed_idx = torch.sort(sorted_idx)
        input_ = pack_padded_sequence(input=input_, lengths=sorted_lengths.data.tolist(), batch_first=True)
        if lstm_hidden is None:
            lstm_out, lstm_hidden = self.lstm(input_)
        else:
            lstm_out, lstm_hidden = self.lstm(lstm_out,lstm_hidden)
        lstm_pad_out = pad_packed_sequence(lstm_out, batch_first=True)[0]
        lstm_pad_out = self.droplstm(lstm_pad_out.transpose(1, 0)).cuda()
        label_embs = label_embs.transpose(1, 0).cuda()
        # lstm_out (batch_size * seq_length * hidden)
        label_attention_output = self.label_attn(lstm_pad_out, label_embs, label_embs)
        # label_attention_output (seq_len, batch_size,embed_size)
        # lstm_out = torch.cat([lstm_pad_out, label_attention_output], -1).transpose(1, 0)
        if self.mode   == "add":
            lstm_out = (lstm_pad_out + label_attention_output).transpose(1, 0)            
            return lstm_out[reversed_idx],label_attention_output.transpose(1, 0)[reversed_idx]
        elif self.mode == "concat":
            lstm_out = torch.cat([lstm_pad_out, label_attention_output], -1).transpose(1, 0)
            return lstm_out[reversed_idx],label_attention_output.transpose(1, 0)[reversed_idx]
        elif self.mode == "mul":
            lstm_out = (lstm_pad_out * label_attention_output).transpose(1, 0)
            return lstm_out[reversed_idx],label_attention_output.transpose(1, 0)[reversed_idx]            
        else:
            return label_attention_output[reversed_idx].transpose(1, 0),label_attention_output.transpose(1, 0)[reversed_idx]

class multihead_attention(nn.Module):

    def __init__(self, num_units, num_heads=1, dropout_rate=0, gpu=True, causality=False):
        '''Applies multihead attention.
        Args:
            num_units: A scalar. Attention size.
            dropout_rate: A floating point number.
            causality: Boolean. If true, units that reference the future are masked.
            num_heads: An int. Number of heads.
        '''
        super(multihead_attention, self).__init__()
        self.gpu = gpu
        self.num_units = num_units
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.causality = causality
        self.Q_proj = nn.Sequential(nn.Linear(self.num_units, self.num_units), nn.ReLU())
        self.K_proj = nn.Sequential(nn.Linear(self.num_units, self.num_units), nn.ReLU())
        self.V_proj = nn.Sequential(nn.Linear(self.num_units, self.num_units), nn.ReLU())
        if self.gpu:
            self.Q_proj = self.Q_proj.cuda()
            self.K_proj = self.K_proj.cuda()
            self.V_proj = self.V_proj.cuda()


        self.output_dropout = nn.Dropout(p=self.dropout_rate)

    def forward(self, queries, keys, values,last_layer = False):
        # keys, values: same shape of [N, T_k, C_k]
        # queries: A 3d Variable with shape of [N, T_q, C_q]
        # Linear projections
        Q = self.Q_proj(queries)  # (N, T_q, C)
        K = self.K_proj(keys)  # (N, T_q, C)
        V = self.V_proj(values)  # (N, T_q, C)
        # Split and concat
        Q_ = torch.cat(torch.chunk(Q, self.num_heads, dim=2), dim=0)  # (h*N, T_q, C/h)
        K_ = torch.cat(torch.chunk(K, self.num_heads, dim=2), dim=0)  # (h*N, T_q, C/h)
        V_ = torch.cat(torch.chunk(V, self.num_heads, dim=2), dim=0)  # (h*N, T_q, C/h)
        # Multiplication
        outputs = torch.bmm(Q_, K_.permute(0, 2, 1))  # (h*N, T_q, T_k)
        # Scale
        outputs = outputs / (K_.size()[-1] ** 0.5)

        # Activation
        if last_layer == False:
            outputs = F.softmax(outputs, dim=-1)  # (h*N, T_q, T_k)
        # Query Masking
        query_masks = torch.sign(torch.abs(torch.sum(queries, dim=-1)))  # (N, T_q)
        query_masks = query_masks.repeat(self.num_heads, 1)  # (h*N, T_q)
        query_masks = torch.unsqueeze(query_masks, 2).repeat(1, 1, keys.size()[1])  # (h*N, T_q, T_k)
        outputs = outputs * query_masks
        # Dropouts
        outputs = self.output_dropout(outputs)  # (h*N, T_q, T_k)
        if last_layer == True:
            return outputs
        # Weighted sum
        outputs = torch.bmm(outputs, V_)  # (h*N, T_q, C/h)
        # Restore shape
        outputs = torch.cat(torch.chunk(outputs, self.num_heads, dim=0), dim=2)  # (N, T_q, C)
        # Residual connection
        outputs += queries

        return outputs
