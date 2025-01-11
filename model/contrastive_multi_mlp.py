import torch
import torch.nn as nn
import os
import numpy as np
import logging

from utils import *
from info_nce import InfoNCE


class ContrastiveMultiMlp(nn.Module):
    def __init__(self,
                 hidden_size,
                 embedding_size=None,
                 hidden_layers=None,
                 dropout=None,
                 in_bn=True,
                 hid_bn=True,
                 out_bn=True,
                 make_ff=True,
                 codebook_size=10,
                 ):
        super().__init__()
        logging.info(f"initialize {self.__class__.__name__}")

        torch.rand(2, 3, 2)

        self.embedding_size = embedding_size
        self.codebook_size = codebook_size

        self.latent_embeddings = torch.rand([codebook_size, self.embedding_size])

        if make_ff:
            self.ffr = self._make_ff(dropout,
                                     self.embedding_size,
                                     hidden_size,
                                     hidden_layers,
                                     in_bn=in_bn,
                                     hid_bn=hid_bn,
                                     out_bn=out_bn)
            self.ffp = self._make_ff(dropout,
                                     384,
                                     [384, 384, 384],
                                     3,
                                     in_bn=True,
                                     hid_bn=True,
                                     out_bn=True,
                                     out=False)
            self.ffq = self._make_ff(dropout,
                                     384,
                                     [384, 384, 384],
                                     3,
                                     in_bn=True,
                                     hid_bn=True,
                                     out_bn=True,
                                     out=False)
            self.ffa = self._make_ff(dropout,
                                     384,
                                     [384, 384, 384],
                                     3,
                                     in_bn=True,
                                     hid_bn=True,
                                     out_bn=True,
                                     out=False)

            # self.att = ScaledDotProductAttention(d_model=self.embedding_size, d_k=self.embedding_size,
            #                                     d_v=self.embedding_size, h=4, dropout=dropout)
            self.att_pre = SimpleConcatAttention(emb_dim=self.embedding_size)
            self.att_qa = SimpleConcatAttention(emb_dim=self.embedding_size)

    def _make_ff(self, dropout, in_size, hidden_size, hidden_layers, in_bn=True, hid_bn=True, out_bn=True, out=True):

        def get_linear(in_size, hidden_size):
            l = torch.nn.Linear(in_size, hidden_size)
            # torch.nn.init.xavier_normal_(l.weight.data, gain=torch.nn.init.calculate_gain('relu'))
            return l

        def get_block(in_size, hidden_size, bn, act=True, drop=True):
            result = [
                torch.nn.BatchNorm1d(in_size) if bn else None,
                torch.nn.Dropout(p=dropout) if drop else None,
                get_linear(in_size, hidden_size),
                torch.nn.ReLU() if act else None,
            ]
            return result

        ff_seq = list()
        ff_seq.extend(get_block(in_size, hidden_size[0], bn=in_bn))
        for i in range(1, hidden_layers): ff_seq.extend(get_block(hidden_size[i - 1], hidden_size[i], bn=hid_bn))
        if out: ff_seq.extend(get_block(hidden_size[-1], 1, bn=out_bn, act=False, drop=False))

        return Sequential(
            *ff_seq
        )

    def forward(self,
                pre_vec,
                pre_num,
                q_vec,
                q_num,
                a_vec,
                a_num):
        """
        For clarity, we drop the operation with different lengths of data, which is used to get more accurate
        representations. Readers can use torch.stack() to compose them.

        q_vec_round
        a_vec_round
        mean_q_vec
        mean_a_vec
        mean_qa_vec
        tilde_pre_vec

        Using q_vec_round as an example, we can apply:
            q_vec_round_list = []
            q_vec_round = self.ffq(q_vec.view(-1, self.embedding_size)).view(batch_size, -1, self.embedding_size)
            for i, length in enumerate(q_num):
                q_vec_round_list.append(q_vec_round[i][:length])
            q_vec_round = torch.stack(q_vec_round_list, dim=0)

        Instead of $bN^{qa}$, we now have $\sum_{i=1}^{B}{N_i^{qa}}$ for QA-level contrastive loss

        Besides, other attention mechanisms (i.e., qkv) can be applied as the substitute for the current version.
        We find different attention get similar results in our experiments.
        :param pre_vec: [b, N^p, d]
        :param pre_num: [b]
        :param q_vec:  [b, N^{qa}, d]
        :param q_num: [b]
        :param a_vec:  [b, N^{qa}, d]
        :param a_num: [b]
        :return:
        """
        batch_size = pre_vec[0]
        pre_vec_sentence = self.ffp(pre_vec.view(-1, self.embedding_size))  # [bN^p, d]

        # QA-level
        q_vec_round = self.ffq(q_vec.view(-1, self.embedding_size))  # [bN^{qa}, d]
        a_vec_round = self.ffa(a_vec.view(-1, self.embedding_size))  # [bN^{qa}, d]

        # Conversation-level
        mean_q_vec = torch.mean(q_vec_round.view(batch_size, -1, self.embedding_size), dim=1)  # [b, d]
        mean_a_vec = torch.mean(a_vec_round.view(batch_size, -1, self.embedding_size), dim=1)  # [b, d]

        # Transcript-level
        mean_qa_vec = mean_q_vec + mean_a_vec  # [b, d]
        topic_aware_p_vec, _ = self.att_pre(self.latent_embeddings,
                                            pre_vec_sentence.view(batch_size, -1, self.embedding_size),
                                            dim=-2)  # [b, K, d] out
        _, topic_att = self.att_qa(self.latent_embeddings,
                                   mean_qa_vec,
                                   dim=-1)  # [K, b] att

        topic_att = topic_att.transpose(0, 1)
        topic_att = topic_att.unsqueeze(-1)  # [b, K, 1]
        tilde_pre_vec = torch.sum(topic_att * topic_aware_p_vec, dim=-2)  # [b, d]

        # risk prediction
        mean_p_vec = torch.mean(pre_vec_sentence.view(batch_size, -1, self.embedding_size), dim=1)  # [b, N^p, d]

        out = self.ffr(mean_qa_vec+mean_p_vec)

        return out, a_vec_round, q_vec_round, mean_a_vec, mean_q_vec, mean_qa_vec, tilde_pre_vec


if __name__ == '__main__':
    a = torch.rand(32, 10, 1)
    b = torch.rand(32, 10, 384)
    c = torch.rand(32, 10, 384)
    att = a*b
    out = torch.sum(att * c, dim=-2)
