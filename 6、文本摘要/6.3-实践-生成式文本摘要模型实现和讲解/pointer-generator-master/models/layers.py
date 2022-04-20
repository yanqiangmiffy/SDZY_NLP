# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence
from torch.nn.utils.rnn import pack_padded_sequence
from utils import config
from models.basic import BasicModule
from models.attention import Attention


class Encoder(BasicModule):
    def __init__(self):
        super(Encoder, self).__init__()
        self.src_word_emb = nn.Embedding(config.vocab_size, config.emb_dim)
        self.lstm = nn.LSTM(config.emb_dim, config.hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(config.hidden_dim * 2, config.hidden_dim * 2, bias=False)
        self.init_params()

    def forward(self, input, seq_lens):
        """

        :param input: 输入序列id
        :param seq_lens: 输入序列长度
        :return:
            encoder_outputs: 每个token的输出向量,[seq_len, bs, 2 * hidden_dim]
            encoder_feature: 在batch上铺平的encoder隐向量
            hidden: 最后时刻的输出[h,c],[2 * layer, bs, hidden_dim]
        """
        embedded = self.src_word_emb(input)  # （bs,len,emb_dim）

        # --
        packed = pack_padded_sequence(embedded, seq_lens, batch_first=True)  # 忽略padding部分，避免额外计算
        output, hidden = self.lstm(packed)  # output

        encoder_outputs, _ = pad_packed_sequence(output, batch_first=True)  # pack_padded_sequence的逆处理
        encoder_outputs = encoder_outputs.contiguous()  # （bs, len, 2*hidden_dim）

        encoder_feature = encoder_outputs.view(-1, 2 * config.hidden_dim)  # （b_s×seq_len,2*hidden_dim）
        encoder_feature = self.fc(encoder_feature)  # （b_s×seq_len, 2 * hidden_dim）, 在bs维度铺平,下面用

        return encoder_outputs, encoder_feature, hidden


class ReduceState(BasicModule):
    """
    实现数据降维
    由于Encoder部分选用的是双向LSTM，而Decoder部分选用的是单向LSTM，因此若直接对Encoder的hidden state与
    Decoder的hidden state进行运算势必会出现维度冲突，因此需要维度降维，这里采用的是简单的将Encoder的双向
    LSTM中两个方向的hidden state简单相加(这里处理稍微复杂版本， 加一层Linear和relu来降维)
    """

    def __init__(self):
        super(ReduceState, self).__init__()

        self.reduce_h = nn.Linear(config.hidden_dim * 2, config.hidden_dim)
        self.reduce_c = nn.Linear(config.hidden_dim * 2, config.hidden_dim)
        self.init_params()

    def forward(self, hidden):
        h, c = hidden  # h, c dim = 2 x b x hidden_dim
        h_in = h.transpose(0, 1).contiguous().view(-1, config.hidden_dim * 2)
        hidden_reduced_h = F.relu(self.reduce_h(h_in))  # (bs, hidden_dim)
        c_in = c.transpose(0, 1).contiguous().view(-1, config.hidden_dim * 2)  # (bs, 2×hidden_dim)
        hidden_reduced_c = F.relu(self.reduce_c(c_in))  # (bs, hidden_dim)
        return (hidden_reduced_h.unsqueeze(0), hidden_reduced_c.unsqueeze(0))  # h, c dim = 1 x b x hidden_dim


class Decoder(BasicModule):
    def __init__(self):
        super(Decoder, self).__init__()
        self.attention_network = Attention()
        # decoder
        self.tgt_word_emb = nn.Embedding(config.vocab_size, config.emb_dim)
        self.con_fc = nn.Linear(config.hidden_dim * 2 + config.emb_dim, config.emb_dim)
        self.lstm = nn.LSTM(config.emb_dim, config.hidden_dim, batch_first=True, bidirectional=False)

        if config.pointer_gen:
            self.p_gen_fc = nn.Linear(config.hidden_dim * 4 + config.emb_dim, 1)

        # p_vocab
        self.fc1 = nn.Linear(config.hidden_dim * 3, config.hidden_dim)
        self.fc2 = nn.Linear(config.hidden_dim, config.vocab_size)

        self.init_params()

    def forward(self, y_t, s_t, enc_out, enc_fea, enc_padding_mask,
                c_t, extra_zeros, enc_batch_extend_vocab, coverage, step):
        """
        y_t:上一个时刻的label(token_id)， (b_s)
        c_t:context_vector，(b_s, 2×hidden_dim)
        s_t: (hidden, cell), decoder_states (第一时刻是encoder的输出)
        enc_out: encoder所有时刻的隐向量 ,[bs,seq_len,hidden_dim * 2]； enc_fea:
        """
        if not self.training and step == 0:
            dec_h, dec_c = s_t
            s_t_hat = torch.cat((dec_h.view(-1, config.hidden_dim),
                                 dec_c.view(-1, config.hidden_dim)), 1)  # B x 2*hidden_dim
            c_t, _, coverage_next = self.attention_network(s_t_hat, enc_out, enc_fea,
                                                           enc_padding_mask, coverage)
            coverage = coverage_next

        y_t_embd = self.tgt_word_emb(y_t)  # （bs，emb_dim）
        x = self.con_fc(torch.cat((c_t, y_t_embd), 1))  # 把c_t和y_t_embd拼接起来强化Encoder输入.  x:当前时刻输入 [bs, hidden_dim],
        lstm_out, s_t = self.lstm(x.unsqueeze(1),
                                  s_t)  # 当前时刻输出,lstm_out:[bs,1,hidden_dim], decoder是单向的. 这里注意:只有一个step, lstm_out=s_t[0]

        dec_h, dec_c = s_t  # 当前时刻h,c: [1, len, hidden_dim],  [1, len, hidden_dim]
        s_t_hat = torch.cat((dec_h.view(-1, config.hidden_dim),
                             dec_c.view(-1, config.hidden_dim)), 1)  # (bt, 2*hidden_dim)，cell结果和隐状态cat起来作为当前时刻的输出向量
        c_t, attn_dist, coverage_next = self.attention_network(s_t_hat, enc_out, enc_fea,
                                                               enc_padding_mask, coverage)  # attn_dist, source的注意力向量

        if self.training or step > 0:
            coverage = coverage_next  # 更新coverage,已经将当前时刻的attention向量加入了

        p_gen = None
        if config.pointer_gen:
            p_gen_inp = torch.cat((c_t, s_t_hat, x),
                                  1)  # B x (2*2*hidden_dim + emb_dim), 上下文向量c_t:[bs, 2*hd], 输出隐向量s_t_hat:[bs, 2*hd], 输入向量x:[bs, ed]
            p_gen = self.p_gen_fc(p_gen_inp)
            p_gen = torch.sigmoid(p_gen)  # 计算p_gen概率,[bs, 1]

        output = torch.cat((lstm_out.view(-1, config.hidden_dim), c_t),
                           1)  # B x hidden_dim * 3, 上下文向量与decoder输出向量concat起来
        output = self.fc1(output)  # B x hidden_dim, 计算在vocab中的概率分布,使用两个fc+softmax

        output = self.fc2(output)  # B x vocab_size
        vocab_dist = F.softmax(output, dim=1)  # 计算token预测概率,[bs,vocab_size]

        if config.pointer_gen:
            vocab_dist_ = p_gen * vocab_dist  # [bs, vocab_size]
            attn_dist_ = (1 - p_gen) * attn_dist  # [bs, sen_len]

            if extra_zeros is not None:
                vocab_dist_ = torch.cat([vocab_dist_, extra_zeros], 1)  # 在字典长度维度(1), 扩充最大oovs数量
            # self_tensor.scatter_add_(dim, index_tensor, other_tensor), 将other_tensor中的数据，按照index_tensor中的索引位置，添加至self_tensor矩阵中
            final_dist = vocab_dist_.scatter_add(1, enc_batch_extend_vocab,
                                                 attn_dist_)  # 计算在扩展字典上的概率分布, 只将在vocab中对应的attn_dist_加进来, OOV加到对应的扩展词典的位置上
        else:
            final_dist = vocab_dist

        return final_dist, s_t, c_t, attn_dist, p_gen, coverage
