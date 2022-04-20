from random import randint, shuffle, choice
from random import random as rand
import math
import torch
import json
import os
import random
from src.loader_utils import get_random_word, batch_list_to_batch_tensors, Pipeline
from torch.utils.data import RandomSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

# Input file format :
# 1. One sentence per line. These should ideally be actual sentences,
#    not entire paragraphs or arbitrary spans of text. (Because we use
#    the sentence boundaries for the "next sentence prediction" task).
# 2. Blank lines between documents. Document boundaries are needed
#    so that the "next sentence prediction" task doesn't span between documents.

TOTAL_SPLIT_TOKENS = ['。', '？', '!', '！', '?', ',', '，', ';', '；']
BIG_SPLIT_TOKENS = ['。', '？', '!', '！', '?', '，']
SMALL_SPLIT_TOKENS = [',', '，', ';', '；']


def truncate_tokens_pair(tokens_a, tokens_b, max_len, max_len_a=0, max_len_b=0, trunc_seg=None,
                         always_truncate_tail=False):
    num_truncated_a = [0, 0]
    num_truncated_b = [0, 0]
    while True:
        if len(tokens_a) + len(tokens_b) <= max_len:
            break
        if (max_len_a > 0) and len(tokens_a) > max_len_a:
            trunc_tokens = tokens_a
            num_truncated = num_truncated_a
        elif (max_len_b > 0) and len(tokens_b) > max_len_b:
            trunc_tokens = tokens_b
            num_truncated = num_truncated_b
        elif trunc_seg:
            # truncate the specified segment
            if trunc_seg == 'a':
                trunc_tokens = tokens_a
                num_truncated = num_truncated_a
            else:
                trunc_tokens = tokens_b
                num_truncated = num_truncated_b
        else:
            # truncate the longer segment
            if len(tokens_a) > len(tokens_b):
                trunc_tokens = tokens_a
                num_truncated = num_truncated_a
            else:
                trunc_tokens = tokens_b
                num_truncated = num_truncated_b
        # whether always truncate source sequences
        if (not always_truncate_tail) and (rand() < 0.5):
            del trunc_tokens[0]
            num_truncated[0] += 1
        else:
            trunc_tokens.pop()
            num_truncated[1] += 1
    return num_truncated_a, num_truncated_b


class Seq2SeqDataset(torch.utils.data.Dataset):
    """ Load sentence pair (sequential or random order) from corpus """

    def __init__(self, data_dir, examples_size_once, tokenizer, max_len, short_sampling_prob=0.1,
                 sent_reverse_order=False, bi_uni_pipeline=[],
                 logger=None):
        super().__init__()
        self.short_sampling_prob = short_sampling_prob
        self.bi_uni_pipeline = bi_uni_pipeline
        self.sent_reverse_order = sent_reverse_order
        self.logger = logger
        self.data_dir = data_dir
        self.examples_size_once = examples_size_once
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.ex_list = self.read_sentence_pairs1()  # 所有句子
        print('ex_list size: %d' % len(self.ex_list))
        # 加载全部数据

    def __len__(self):
        return len(self.ex_list)

    def __getitem__(self, idx):
        instance = self.ex_list[idx]
        proc = choice(self.bi_uni_pipeline)
        instance = proc(instance)
        return instance

    def read_sentence_pairs1(self):
        """
        加载self.data_dir文件的数据，构造所有输入sentence
        :return:
        """
        self.logger.info('start read sentence pair...')
        sentence_pairs = []
        if os.path.exists(self.data_dir):
            with open(self.data_dir, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                for document in tqdm(lines):
                    for tokens_a, tokens_b in self.create_sentence_pairs(document.strip()):
                        sentence_pairs.append((tokens_a, tokens_b))
                    if len(sentence_pairs) <= self.examples_size_once or self.examples_size_once <= 0:
                        continue
                    else:
                        break
        random.shuffle(sentence_pairs)
        self.logger.info('read sentence pair finished!')
        return sentence_pairs

    def read_sentence_pairs(self):
        """
        从self.data_dir中读取self.file_size个样本
        :return: list
        """
        sentence_pairs = []
        file_names = [os.path.join(self.data_dir, file_name) for file_name in os.listdir(self.data_dir)]
        file_index = 0
        while file_index < len(file_names):
            file_name = file_names[file_index]
            with open(file_name, 'r', encoding='utf-8') as f:
                for document in f:
                    for tokens_a, tokens_b in self.create_sentence_pairs(document.strip()):
                        sentence_pairs.append((tokens_a, tokens_b))
                        if len(sentence_pairs) == self.examples_size_once:
                            yield sentence_pairs
                            sentence_pairs = []
            if file_index == len(file_names) - 1:
                file_index = 0

    def create_sentence_pairs(self, document):
        # 输入文档，按sentence分割，选择大于max_len的句子组合，然后将此组合【随机位置】分割成一个连续的文本对儿
        sentences = []
        sen = []
        for p, w in enumerate(document):
            sen.append(w)
            if w in BIG_SPLIT_TOKENS or p == len(document) - 1:
                if len(sen) > 1:
                    tokens = self.tokenizer.tokenize(''.join(sen))
                    if tokens:
                        sentences.append(tokens)
                sen = []
        i = 0
        current_chunk = []
        current_length = 0

        while i < len(sentences):
            segment = sentences[i]
            current_chunk.append(segment)
            current_length += len(segment)
            if i == len(sentences) - 1 or current_length >= self.max_len:
                if len(current_chunk) == 1: continue
                if current_chunk:
                    a_end = 1
                    if len(current_chunk) >= 2:
                        a_end = random.randint(1, len(current_chunk) - 1)
                    tokens_a = []
                    tokens_b = []
                    for j in range(a_end):
                        tokens_a.extend(current_chunk[j])
                    for j in range(a_end, len(current_chunk)):
                        tokens_b.extend(current_chunk[j])
                    assert len(tokens_a) >= 1
                    assert len(tokens_b) >= 1
                    yield (tokens_a, tokens_b)
                    current_chunk = []
                    current_length = 0
            i += 1


class TrainDataLoader(object):
    def __init__(self, bi_uni_pipline, world_size, train_batch_size, num_workers, examples_size_once, data_dir,
                 tokenizer, max_len, rank, logger=None):
        self.bi_uni_pipeline = bi_uni_pipline
        self.world_size = world_size
        self.examples_size_once = examples_size_once  # DataLoader每次加载的数据量
        self.train_batch_size = train_batch_size
        self.num_workers = num_workers
        self.data_dir = data_dir
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.logger = logger
        self.train_dataset = Seq2SeqDataset(data_dir,
                                            examples_size_once=examples_size_once,
                                            tokenizer=tokenizer,
                                            max_len=max_len,
                                            bi_uni_pipeline=bi_uni_pipline,
                                            logger=self.logger)
        if self.world_size < 1:
            self.train_sampler = RandomSampler(self.train_dataset, replacement=False)
        else:
            self.train_sampler = DistributedSampler(self.train_dataset, num_replicas=world_size, rank=rank)

        self.train_dataloader = torch.utils.data.DataLoader(self.train_dataset,
                                                            batch_size=train_batch_size,
                                                            sampler=self.train_sampler,
                                                            num_workers=self.num_workers,
                                                            collate_fn=batch_list_to_batch_tensors,
                                                            pin_memory=True)

    def __iter__(self):
        for batch in self.train_dataloader:
            yield batch

