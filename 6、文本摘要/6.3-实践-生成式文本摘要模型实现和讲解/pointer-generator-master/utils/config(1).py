# -*- coding: utf-8 -*-

SENTENCE_STA = '<s>'
SENTENCE_END = '</s>'

UNK = 0
PAD = 1
BOS = 2
EOS = 3

PAD_TOKEN = '[PAD]'
UNK_TOKEN = '[UNK]'
BOS_TOKEN = '[BOS]'
EOS_TOKEN = '[EOS]'

beam_size = 4
emb_dim = 32
batch_size = 8
hidden_dim = 64
max_enc_steps = 60  # source最大输入长度
max_dec_steps = 15  # 解码最大长度
max_tes_steps = 10
min_dec_steps = 2
vocab_size = 7000   # 共7612个词
epoch = 1
save_iter = 20
flush_iter = 10
lr = 0.15
cov_loss_wt = 1.0
pointer_gen = True
is_coverage = True
lr_decay_rate = 0.95
max_grad_norm = 2.0
adagrad_init_acc = 0.1
rand_unif_init_mag = 0.02
trunc_norm_init_std = 1e-4

eps = 1e-12
use_gpu = True
lr_coverage = 0.15
max_iterations = 50

log_root = '/Users/himon/Jobs/class/project-class/class3/code/pointer-generator-master/logs'
train_data_path = "/Users/himon/Jobs/class/project-class/class3/code/pointer-generator-master/lcsta_data/train_sample.txt"
eval_data_path = '/Users/himon/Jobs/class/project-class/class3/code/pointer-generator-master/lcsta_data/dev_sample.txt'
vocab_path = '/Users/himon/Jobs/class/project-class/class3/code/pointer-generator-master/lcsta_data/vocab.txt'
decode_data_path = '/Users/himon/Jobs/class/project-class/class3/code/pointer-generator-master/lcsta_data/samples_gen_sam.txt.txt'
