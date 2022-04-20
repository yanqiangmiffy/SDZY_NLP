# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import config
from models.basic import BasicModule


class Attention(BasicModule):
    def __init__(self):
        super(Attention, self).__init__()

        self.fc = nn.Linear(config.hidden_dim * 2, 1, bias=False)
        self.dec_fc = nn.Linear(config.hidden_dim * 2, config.hidden_dim * 2)
        if config.is_coverage:
            self.con_fc = nn.Linear(1, config.hidden_dim * 2, bias=False)

        self.init_params()

    def forward(self, s_t, enc_out, enc_fea, enc_padding_mask, coverage):
        """
        计算attention向量和context vector
        这里attention的实现: 相加(解码器输出隐状态,编码器输出隐状态,coverage状态向量) -> tanh -> fc -> softmax -> bmm
        -general方式的改版.
        s_t: decoder_state, cell和state拼起来的
        enc_out:；
        coverage:[bs, sent_len];
        """
        b, l, n = list(enc_out.size())  #

        dec_fea = self.dec_fc(s_t)  # 解码器输出隐状态, 过一层linear
        dec_fea_expanded = dec_fea.unsqueeze(1).expand(b, l, n).contiguous()  # [bs, seq_len, 2*hidden_dim]
        dec_fea_expanded = dec_fea_expanded.view(-1, n)     # 铺平, [bs*seq_len , 2*hidden_dim]

        att_features = enc_fea + dec_fea_expanded           # 解码器输出隐状态 + 编码器输出隐状态, [bs*seq_len , 2*hidden_dim]
        if config.is_coverage:
            coverage_inp = coverage.view(-1, 1)             # 在bs上铺平, [bs*seq_len , 1]
            coverage_fea = self.con_fc(coverage_inp)        # coverage状态向量扩维, [bs*seq_len, 2*hidden_dim]
            att_features = att_features + coverage_fea      # + coverage状态向量

        e = torch.tanh(att_features)                        # 加入非线性激活函数tanh, [bs*seq_len , 2*hidden_dim]
        scores = self.fc(e)                                 # [bs*seq_len , 1]
        scores = scores.view(-1, l)                         # [bs, seq_len]
        attn_dist_ = F.softmax(scores, dim=1) * enc_padding_mask  # [bs, seq_len]
        normalization_factor = attn_dist_.sum(1, keepdim=True)    # 上面mask了需要重新归一化
        attn_dist = attn_dist_ / normalization_factor

        # 计算context vector
        attn_dist = attn_dist.unsqueeze(1)                        # [bs, 1, seq_len]
        c_t = torch.bmm(attn_dist, enc_out)                       # 两个tensor的矩阵乘法, [bs,1,seq_len] * [bs,seq_len,hidden_dim] ,
        c_t = c_t.view(-1, config.hidden_dim * 2)                 # [bs , 2*hidden_dim], context vector

        attn_dist = attn_dist.view(-1, l)                         # [bs, seq_len]

        if config.is_coverage:
            coverage = coverage.view(-1, l)
            coverage = coverage + attn_dist     # coverage=过去所有时刻的attention分布的和

        return c_t, attn_dist, coverage

