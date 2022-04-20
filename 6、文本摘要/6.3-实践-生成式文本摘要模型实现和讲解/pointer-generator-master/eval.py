# -*- coding: utf-8 -*-

from __future__ import unicode_literals, print_function, division
import os
import time
import torch
from torch.utils.tensorboard import SummaryWriter
from models.model import Model
from utils import config
from utils.dataset import Vocab
from utils.dataset import Batcher
from utils.utils import get_input_from_batch
from utils.utils import get_output_from_batch
from utils.utils import calc_running_avg_loss

use_cuda = config.use_gpu and torch.cuda.is_available()


class Evaluate(object):
    def __init__(self, model_path):
        self.vocab = Vocab(config.vocab_path, config.vocab_size)
        self.batcher = Batcher(data_path=config.eval_data_path, vocab=self.vocab, mode='eval',
                               batch_size=config.batch_size, single_pass=True,
                               max_iterations=config.max_iterations)
        time.sleep(15)
        model_name = os.path.basename(model_path)

        eval_dir = os.path.join(config.log_root, 'eval_%s' % (model_name))
        if not os.path.exists(eval_dir):
            os.mkdir(eval_dir)
        self.summary_writer = SummaryWriter(eval_dir)

        self.model = Model(model_path, is_eval=True)

    def eval_one_batch(self, batch):
        enc_batch, dec_lens_var, enc_pos, enc_padding_mask, enc_batch_extend_vocab, \
        extra_zeros, c_t, coverage = \
            get_input_from_batch(batch, use_cuda)
        dec_batch, dec_lens, dec_pos, dec_padding_mask, max_dec_len, tgt_batch = \
            get_output_from_batch(batch, use_cuda)

        enc_out, enc_fea, enc_h = self.model.encoder(enc_batch, dec_lens_var)
        s_t = self.model.reduce_state(enc_h)

        step_losses = []
        for di in range(min(max_dec_len, config.max_dec_steps)):
            y_t = dec_batch[:, di]  # Teacher forcing
            final_dist, s_t, c_t, attn_dist, p_gen, next_coverage = self.model.decoder(y_t, s_t,
                                                                                       enc_out, enc_fea,
                                                                                       enc_padding_mask, c_t,
                                                                                       extra_zeros,
                                                                                       enc_batch_extend_vocab, coverage,
                                                                                       di)
            tgt = tgt_batch[:, di]
            gold_probs = torch.gather(final_dist, 1, tgt.unsqueeze(1)).squeeze()
            step_loss = -torch.log(gold_probs + config.eps)
            if config.is_coverage:
                step_coverage_loss = torch.sum(torch.min(attn_dist, coverage), 1)
                step_loss = step_loss + config.cov_loss_wt * step_coverage_loss
                coverage = next_coverage

            step_mask = dec_padding_mask[:, di]
            step_loss = step_loss * step_mask
            step_losses.append(step_loss)

        sum_step_losses = torch.sum(torch.stack(step_losses, 1), 1)
        sum_step_losses = sum_step_losses.cpu()
        batch_avg_loss = sum_step_losses / dec_lens_var
        loss = torch.mean(batch_avg_loss)

        return loss.data[0]

    def run(self):
        start = time.time()
        running_avg_loss, iter = 0, 0
        batch = self.batcher.next_batch()
        print_interval = 100
        while batch is not None:
            loss = self.eval_one_batch(batch)
            running_avg_loss = calc_running_avg_loss(loss, running_avg_loss, self.summary_writer, iter)
            iter += 1

            if iter % print_interval == 0:
                self.summary_writer.flush()
                print('step: %d, second: %.2f , loss: %f' % (iter, time.time() - start, running_avg_loss))
                start = time.time()
            batch = self.batcher.next_batch()

        return running_avg_loss


if __name__ == '__main__':
    # model_filename = sys.argv[1]
    model_filename = '/home/humeng/e/projects/hot_search/modules/pointer-generator-master/logs/train_1611718565/models/model_34000'

    eval_processor = Evaluate(model_filename)
    eval_processor.run()
