import numpy as np
import torch
from torch import nn
from torch.nn import init
import torch.nn.functional as F


class SimpleConcatAttention(nn.Module):
    '''
    '''

    def __init__(self, emb_dim):
        super(SimpleConcatAttention, self).__init__()
        self.emb_dim = emb_dim
        self.W = nn.Parameter(torch.Tensor(emb_dim, emb_dim))

    def forward(self, query, key, dim=-2, att_mask=None):
        """
        :param query: [b, K, d]
        :param key: [b, N, d]
        :param dim: can be -2 or -1, we find that use both -1 and -2 get the similar result
        :param att_mask:
        :return:
        """
        out = None
        att = None
        if len(query.shape) == len(key.shape) == 2:
            att = query @ self.W @ key.transpose(0, 1) / np.sqrt(self.emb_dim)
            if att_mask is not None:
                att = att.masked_fill(~att_mask, -np.inf)
            att = F.softmax(att, dim=dim)
            out = att @ key
        elif (len(query.shape) == 2) and (len(key.shape) == 3):
            batch_size = key.shape[0]
            query = query.unsqueeze(dim=0)
            query = query.tile(batch_size, 1, 1)  # [b, K, d]
            query = query @ self.W / np.sqrt(self.emb_dim)
            att = torch.bmm(query, key.transpose(1, 2))  # [b, K, N]
            if att_mask is not None:
                att = att.masked_fill(~att_mask, -np.inf)
            att = F.softmax(att, dim=dim)
            out = torch.bmm(att, key)
        else:
            raise NotImplementedError
        return out, att


class ScaledDotProductAttention(nn.Module):
    '''
    Scaled dot-product attention
    '''

    def __init__(self, d_model, d_k, d_v, h, dropout=.1):
        '''
        :param d_model: Output dimensionality of the model
        :param d_k: Dimensionality of queries and keys
        :param d_v: Dimensionality of values
        :param h: Number of heads
        '''
        super(ScaledDotProductAttention, self).__init__()
        self.fc_q = nn.Linear(d_model, h * d_k)
        self.fc_k = nn.Linear(d_model, h * d_k)
        self.fc_v = nn.Linear(d_model, h * d_v)
        self.fc_o = nn.Linear(h * d_v, d_model)
        self.dropout = nn.Dropout(dropout)

        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.h = h

        self.layer_norm = nn.LayerNorm(d_v)

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, queries, keys, values, attention_mask=None, attention_weights=None, dim=-2):
        '''
        Computes
        :param queries: Queries (b_s, nq, d_model)
        :param keys: Keys (b_s, nk, d_model)
        :param values: Values (b_s, nk, d_model)
        :param attention_mask: Mask over attention values (b_s, h, nq, nk). True indicates masking.
        :param attention_weights: Multiplicative weights for attention values (b_s, h, nq, nk).
        :param dim: Used for attention softmax. Can be -1 or -2, determined by the specific task.
        :return:
        '''
        residual = queries

        b_s, nq = queries.shape[:2]
        nk = keys.shape[1]

        q = self.fc_q(queries).view(b_s, nq, self.h, self.d_k).permute(0, 2, 1, 3)  # (b_s, h, nq, d_k)
        k = self.fc_k(keys).view(b_s, nk, self.h, self.d_k).permute(0, 2, 3, 1)  # (b_s, h, d_k, nk)
        v = self.fc_v(values).view(b_s, nk, self.h, self.d_v).permute(0, 2, 1, 3)  # (b_s, h, nk, d_v)

        att = torch.matmul(q, k) / np.sqrt(self.d_k)  # (b_s, h, nq, nk)
        if attention_weights is not None:
            att = att * attention_weights
        if len(attention_mask.size()) != len(att.size()):
            attention_mask = attention_mask.unsqueeze(dim=1).expand(-1, self.h, -1, -1)
        if attention_mask is not None:
            att = att.masked_fill(~attention_mask, -np.inf)
        att = F.softmax(att, dim)

        out = torch.matmul(att, v).permute(0, 2, 1, 3).contiguous().view(b_s, nq, self.h * self.d_v)  # (b_s, nq, h*d_v)
        out = self.fc_o(out)  # (b_s, nq, d_model)
        out = self.layer_norm(residual + out)

        return out


if __name__ == '__main__':
    att = SimpleConcatAttention(emb_dim=384)
    q = torch.rand(10, 384)
    k = torch.rand(32, 100, 384)
    x = att(q, k)

    q = torch.rand(10, 384)
    k = torch.rand(32, 384)
    x = att(q, k)
