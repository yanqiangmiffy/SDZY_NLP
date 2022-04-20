# -*- coding: utf-8 -*-

from __future__ import unicode_literals, print_function, division

import os
import time
import argparse
import os.path
import torch
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
from torch.utils.tensorboard import SummaryWriter
from models.model import Model
from utils import config
from utils.dataset import Vocab
from utils.dataset import Batcher
from utils.utils import get_input_from_batch
from utils.utils import get_output_from_batch
from utils.utils import calc_running_avg_loss

use_cuda = config.use_gpu and torch.cuda.is_available()


class Train(object):
    def __init__(self):
        self.vocab = Vocab(config.vocab_path, config.vocab_size)
        self.batcher = Batcher(self.vocab, config.train_data_path,
                               config.batch_size, single_pass=False, mode='train',
                               max_iterations=config.max_iterations)
        self.save_iter = config.save_iter

        train_dir = os.path.join(config.log_root, 'train_%d' % (int(time.time())))
        if not os.path.exists(train_dir):
            os.mkdir(train_dir)

        self.model_dir = os.path.join(train_dir, 'models')
        if not os.path.exists(self.model_dir):
            os.mkdir(self.model_dir)

        self.summary_writer = SummaryWriter(train_dir)

    def save_model(self, running_avg_loss, iter):
        state = {
            'iter': iter,
            'encoder_state_dict': self.model.encoder.state_dict(),
            'decoder_state_dict': self.model.decoder.state_dict(),
            'reduce_state_dict': self.model.reduce_state.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'current_loss': running_avg_loss
        }
        last_state_dict = os.path.join(self.model_dir, 'model_%d' % (iter - self.save_iter))
        if os.path.isfile(last_state_dict):
            os.remove(last_state_dict)
        model_save_path = os.path.join(self.model_dir, 'model_%d' % (iter))
        torch.save(state, model_save_path)

    def setup_train(self, model_path=None):
        """
        初始化模型,优化器(如果model_path存在,加载并恢复,然后返回开始布数和loss)等
        """
        t_total = config.epoch * self.batcher._batch_queue.qsize()
        self.model = Model(model_path)
        self.initial_lr = config.lr_coverage if config.is_coverage else config.lr

        param_optimizer = list(self.model.encoder.parameters()) + list(self.model.decoder.parameters()) + \
                          list(self.model.reduce_state.parameters())

        self.optimizer = optim.Adagrad(param_optimizer, lr=self.initial_lr,
                                       initial_accumulator_value=config.adagrad_init_acc)

        start_iter, start_loss = 0, 0

        if model_path is not None:
            state = torch.load(model_path, map_location=lambda storage, location: storage)
            start_iter = state['iter']
            start_loss = state['current_loss']

            if not config.is_coverage:
                self.optimizer.load_state_dict(state['optimizer'])
                if use_cuda:
                    for state in self.optimizer.state.values():
                        for k, v in state.items():
                            if torch.is_tensor(v):
                                state[k] = v.cuda()

        return start_iter, start_loss

    def train_one_batch(self, batch, iter):
        enc_batch, enc_lens, enc_pos, enc_padding_mask, enc_batch_extend_vocab, \
        extra_zeros, c_t, coverage = get_input_from_batch(batch, use_cuda)

        dec_batch, dec_lens, dec_pos, dec_padding_mask, max_dec_len, tgt_batch = \
            get_output_from_batch(batch, use_cuda)

        # 清空梯度
        self.optimizer.zero_grad()
        # 执行Encoder
        enc_out, enc_fea, enc_h = self.model.encoder(enc_batch, enc_lens)

        s_t = self.model.reduce_state(enc_h)  # encoder最后时刻降维, [(1, bs, hidden_dim), (1, bs, hidden_dim)]

        step_losses, cove_losses = [], []
        for di in range(min(max_dec_len, config.max_dec_steps)):
            y_t = dec_batch[:, di]  # Teacher forcing
            final_dist, s_t, c_t, attn_dist, p_gen, next_coverage = \
                self.model.decoder(y_t, s_t, enc_out, enc_fea, enc_padding_mask, c_t,
                                   extra_zeros, enc_batch_extend_vocab, coverage, di)
            tgt = tgt_batch[:, di]  # 取当前step的解码token_id
            step_mask = dec_padding_mask[:, di]
            gold_probs = torch.gather(final_dist, 1, tgt.unsqueeze(1)).squeeze()  # 从原tensor中获取指定dim和指定index的数据, [bs, ]
            step_loss = -torch.log(gold_probs + config.eps)  # 计算loss(负对数似然)
            if config.is_coverage:
                step_coverage_loss = torch.sum(torch.min(attn_dist, coverage),
                                               1)  # 计算coverage loss, sum(min(atten_dist, coverage))
                step_loss = step_loss + config.cov_loss_wt * step_coverage_loss  # 总loss
                cove_losses.append(step_coverage_loss * step_mask)
                coverage = next_coverage  # 当前step的coverage传递下去, 更新当前coverage

            step_loss = step_loss * step_mask
            step_losses.append(step_loss)

        sum_losses = torch.sum(torch.stack(step_losses, 1), 1)
        batch_avg_loss = sum_losses / dec_lens
        loss = torch.mean(batch_avg_loss)
        loss.backward()     # 使用平均loss，计算梯度
        self.summary_writer.add_scalar('Train/lr', self.optimizer.param_groups[0]['lr'], iter)

        # 梯度剪裁
        # --
        encoder_parameters = clip_grad_norm_(self.model.encoder.parameters(), config.max_grad_norm)
        decoder_parameters = clip_grad_norm_(self.model.decoder.parameters(), config.max_grad_norm)
        reduce_state_parameters = clip_grad_norm_(self.model.reduce_state.parameters(), config.max_grad_norm)

        assert encoder_parameters >= 0, 'encounter nan in gradients.'
        assert decoder_parameters >= 0, 'encounter nan in gradients.'
        assert reduce_state_parameters >= 0, 'encounter nan in gradients.'

        self.optimizer.step()   #梯度更新

        if config.is_coverage:
            cove_losses = torch.sum(torch.stack(cove_losses, 1), 1)
            batch_cove_loss = cove_losses / dec_lens
            batch_cove_loss = torch.mean(batch_cove_loss)
            return loss.item(), batch_cove_loss.item()

        return loss.item(), 0.

    def run(self, n_iters, model_path=None):
        iter, running_avg_loss = self.setup_train(model_path)
        start = time.time()
        for epoch in range(config.epoch):
            while iter < n_iters:
                batch = self.batcher.next_batch()
                if not batch:
                    break
                loss, cove_loss = self.train_one_batch(batch, iter)

                running_avg_loss = calc_running_avg_loss(loss, running_avg_loss, self.summary_writer, iter)
                iter += 1
                self.summary_writer.add_scalar("train/cov_loss", cove_loss, iter)
                if iter % config.flush_iter == 0:
                    self.summary_writer.flush()
                    print(
                        'epoch:%d, step: %d, second: %.2f , loss: %f, cover_loss: %f' % (
                            epoch, iter, time.time() - start, loss, cove_loss))
                    start = time.time()
                if iter % self.save_iter == 0:
                    self.save_model(running_avg_loss, iter)

            # init batcher,每个epoch都需要重新初始化Batcher来加载全部数据
            print('re-init batcher...')
            self.batcher = Batcher(self.vocab, config.train_data_path,
                                   config.batch_size, single_pass=False, mode='train',
                                   max_iterations=config.max_iterations)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train script")
    parser.add_argument("-m",
                        dest="model_path",
                        required=False,
                        default=None,
                        help="Model file for retraining (default: None).")
    args = parser.parse_args()

    train_processor = Train()
    train_processor.run(config.max_iterations * config.epoch, args.model_path)
