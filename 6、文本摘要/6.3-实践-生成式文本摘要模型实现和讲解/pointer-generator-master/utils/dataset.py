# -*- coding: utf-8 -*-

import csv
import glob
import time
import queue
import struct
import numpy as np
import tensorflow as tf
from random import shuffle
from threading import Thread
from tensorflow.core.example import example_pb2
from utils import utils
from utils import config
import random

random.seed(1234)

# <s> and </s> are used in the data files to segment the abstracts into sentences. They don't receive vocab ids.
SENTENCE_STA = '<s>'
SENTENCE_END = '</s>'

PAD_TOKEN = '[PAD]'  # This has a vocab id, which is used to pad the encoder input, decoder input and target sequence
UNK_TOKEN = '[UNK]'  # This has a vocab id, which is used to represent out-of-vocabulary words
BOS_TOKEN = '[BOS]'  # This has a vocab id, which is used at the start of every decoder input sequence
EOS_TOKEN = '[EOS]'  # This has a vocab id, which is used at the end of untruncated target sequences


# Note: none of <s>, </s>, [PAD], [UNK], [START], [STOP] should appear in the vocab file.


class Vocab(object):

    def __init__(self, file, max_size):
        self.word2idx = {}
        self.idx2word = {}
        self.count = 0  # keeps track of total number of words in the Vocab

        # [UNK], [PAD], [BOS] and [EOS] get the ids 0,1,2,3.
        for w in [UNK_TOKEN, PAD_TOKEN, BOS_TOKEN, EOS_TOKEN]:
            self.word2idx[w] = self.count
            self.idx2word[self.count] = w
            self.count += 1

        # Read the vocab file and add words up to max_size
        with open(file, 'r') as fin:
            for line in fin:
                items = line.split()
                w = items[0]
                if w in [SENTENCE_STA, SENTENCE_END, UNK_TOKEN, PAD_TOKEN, BOS_TOKEN, EOS_TOKEN]:
                    raise Exception(
                        '<s>, </s>, [UNK], [PAD], [BOS] and [EOS] shouldn\'t be in the vocab file, but %s is' % w)
                if w in self.word2idx:
                    raise Exception('Duplicated word in vocabulary file: %s' % w)
                self.word2idx[w] = self.count
                self.idx2word[self.count] = w
                self.count += 1
                if max_size != 0 and self.count >= max_size:
                    break
        print("Finished constructing vocabulary of %i total words. Last word added: %s" % (
            self.count, self.idx2word[self.count - 1]))

    def word2id(self, word):
        if word not in self.word2idx:
            return self.word2idx[UNK_TOKEN]
        return self.word2idx[word]

    def id2word(self, word_id):
        if word_id not in self.idx2word:
            raise ValueError('Id not found in vocab: %d' % word_id)
        return self.idx2word[word_id]

    def size(self):
        return self.count

    def write_metadata(self, path):
        print("Writing word embedding metadata file to %s..." % (path))
        with open(path, "w") as f:
            fieldnames = ['word']
            writer = csv.DictWriter(f, delimiter="\t", fieldnames=fieldnames)
            for i in range(self.size()):
                writer.writerow({"word": self.idx2word[i]})


class Example(object):

    def __init__(self, article, abstract_sentences, vocab):
        # Get ids of special tokens
        bos_decoding = vocab.word2id(BOS_TOKEN)
        eos_decoding = vocab.word2id(EOS_TOKEN)

        # Process the article
        if type(article) == bytes:
            article_words = article.decode().split()
        else:
            article_words = article.split()
        if len(article_words) > config.max_enc_steps:
            article_words = article_words[:config.max_enc_steps]
        self.enc_len = len(article_words)  # store the length after truncation but before padding
        self.enc_inp = [vocab.word2id(w) for w in
                        article_words]  # list of word ids; OOVs are represented by the id for UNK token

        # Process the abstract
        if type(article) == bytes:
            abstract = ' '.encode().join(abstract_sentences).decode()
            abstract_words = abstract.split()  # list of strings
        else:
            abstract = abstract_sentences
            abstract_words = abstract_sentences[0].split()  # list of strings
        abs_ids = [vocab.word2id(w) for w in
                   abstract_words]  # list of word ids; OOVs are represented by the id for UNK token

        # Get the decoder input sequence and target sequence, 把开始标志[BOS]加入source, 结束标志[EOS]加入tgt
        self.dec_inp, self.tgt = self.get_dec_seq(abs_ids, config.max_dec_steps, bos_decoding, eos_decoding)
        self.dec_len = len(self.dec_inp)

        # If using pointer-generator mode, we need to store some extra info
        if config.pointer_gen:
            # enc_inp_extend_vocab:article中所有token的id,如果在词典中,就是词典中id,如果是oov,就是在词典数量基础上扩展的id
            # article_oovs:article中所有oov
            self.enc_inp_extend_vocab, self.article_oovs = utils.article2ids(article_words, vocab)

            # Get a verison of the reference summary where in-article OOVs are represented by their temporary article OOV id
            # 摘要的
            abs_ids_extend_vocab = utils.abstract2ids(abstract_words, vocab, self.article_oovs)

            # Overwrite decoder target sequence so it uses the temp article OOV ids, 构造tgt的下表,包括OOV词语的id
            _, self.tgt = self.get_dec_seq(abs_ids_extend_vocab, config.max_dec_steps, bos_decoding, eos_decoding)

        # Store the original strings
        self.original_article = article
        self.original_abstract = abstract
        self.original_abstract_sents = abstract_sentences

    def get_dec_seq(self, sequence, max_len, start_id, stop_id):
        src = [start_id] + sequence[:]
        tgt = sequence[:]
        if len(src) > max_len:  # truncate
            src = src[:max_len]
            tgt = tgt[:max_len]  # no end_token
        else:  # no truncation
            tgt.append(stop_id)  # end token
        assert len(src) == len(tgt)
        return src, tgt

    def pad_enc_seq(self, max_len, pad_id):
        while len(self.enc_inp) < max_len:
            self.enc_inp.append(pad_id)
        if config.pointer_gen:
            while len(self.enc_inp_extend_vocab) < max_len:
                self.enc_inp_extend_vocab.append(pad_id)

    def pad_dec_seq(self, max_len, pad_id):
        while len(self.dec_inp) < max_len:
            self.dec_inp.append(pad_id)
        while len(self.tgt) < max_len:
            self.tgt.append(pad_id)


class Batch(object):
    def __init__(self, example_list, vocab, batch_size):
        self.batch_size = batch_size
        self.pad_id = vocab.word2id(PAD_TOKEN)  # id of the PAD token used to pad sequences
        self.init_encoder_seq(example_list)  # initialize the input to the encoder
        self.init_decoder_seq(example_list)  # initialize the input and targets for the decoder
        self.store_orig_strings(example_list)  # store the original strings

    def init_encoder_seq(self, example_list):
        """

        :param example_list: 一个batch的Example实例
        :return:
        """
        # Determine the maximum length of the encoder input sequence in this batch
        max_enc_seq_len = max([ex.enc_len for ex in example_list])

        # Pad the encoder input sequences up to the length of the longest sequence
        for ex in example_list:
            ex.pad_enc_seq(max_enc_seq_len, self.pad_id)

        # Initialize the numpy arrays
        # Note: our enc_batch can have different length (second dimension) for each batch because we use dynamic_rnn for the encoder.
        self.enc_batch = np.zeros((self.batch_size, max_enc_seq_len), dtype=np.int32)
        self.enc_lens = np.zeros((self.batch_size), dtype=np.int32)
        self.enc_padding_mask = np.zeros((self.batch_size, max_enc_seq_len), dtype=np.float32)

        # Fill in the numpy arrays
        for i, ex in enumerate(example_list):
            self.enc_batch[i, :] = ex.enc_inp[:]
            self.enc_lens[i] = ex.enc_len
            for j in range(ex.enc_len):
                self.enc_padding_mask[i][j] = 1

        # For pointer-generator mode, need to store some extra info
        if config.pointer_gen:
            # 本batch中source中OOVS的最大数量
            self.max_art_oovs = max([len(ex.article_oovs) for ex in example_list])
            # Store the in-article OOVs themselves
            self.art_oovs = [ex.article_oovs for ex in example_list]
            # Store the version of the enc_batch that uses the article OOV ids, 所有
            self.enc_batch_extend_vocab = np.zeros((self.batch_size, max_enc_seq_len), dtype=np.int32)
            for i, ex in enumerate(example_list):
                self.enc_batch_extend_vocab[i, :] = ex.enc_inp_extend_vocab[:]

    def init_decoder_seq(self, example_list):
        # Pad the inputs and targets
        for ex in example_list:
            ex.pad_dec_seq(config.max_dec_steps, self.pad_id)

        # Initialize the numpy arrays.
        self.dec_batch = np.zeros((self.batch_size, config.max_dec_steps), dtype=np.int32)
        self.tgt_batch = np.zeros((self.batch_size, config.max_dec_steps), dtype=np.int32)
        self.dec_padding_mask = np.zeros((self.batch_size, config.max_dec_steps), dtype=np.float32)
        self.dec_lens = np.zeros((self.batch_size), dtype=np.int32)

        # Fill in the numpy arrays
        for i, ex in enumerate(example_list):
            self.dec_batch[i, :] = ex.dec_inp[:]
            self.tgt_batch[i, :] = ex.tgt[:]
            self.dec_lens[i] = ex.dec_len
            for j in range(ex.dec_len):
                self.dec_padding_mask[i][j] = 1

    def store_orig_strings(self, example_list):
        self.original_articles = [ex.original_article for ex in example_list]  # list of lists
        self.original_abstracts = [ex.original_abstract for ex in example_list]  # list of lists
        self.original_abstracts_sents = [ex.original_abstract_sents for ex in example_list]  # list of list of lists


class Batcher(object):
    def __init__(self, vocab, data_path, batch_size, single_pass, mode, max_iterations, is_test=False):
        self._vocab = vocab
        self._data_path = data_path
        self.batch_size = batch_size
        self.single_pass = single_pass
        self.mode = mode
        self.is_test = is_test
        self.BATCH_QUEUE_MAX = max_iterations
        # Initialize a queue of Batches waiting to be used, and a queue of Examples waiting to be batched
        self._batch_queue = queue.Queue()
        self._example_queue = queue.Queue()

        # Different settings depending on whether we're in single_pass mode or not
        if single_pass:
            self._num_example_q_threads = 1  # just one thread, so we read through the dataset just once
            self._num_batch_q_threads = 1  # just one thread to batch examples
            self._bucketing_cache_size = 1  # only load one batch's worth of examples before bucketing
            self._finished_reading = False  # this will tell us when we're finished reading the dataset
        else:
            self._num_example_q_threads = 1  # num threads to fill example queue
            self._num_batch_q_threads = 1  # num threads to fill batch queue
            self._bucketing_cache_size = 1  # how many batches-worth of examples to load into cache before bucketing

        # Start the threads that load the queues
        self._example_q_threads = []
        self.fill_example_queue()
        self.fill_batch_queue()
        self._batch_q_threads = []

    def next_batch(self):
        """
        取一个batch的数据
        :return:
        """
        # If the batch queue is empty, print a warning
        if self._batch_queue.qsize() == 0:
            tf.compat.v1.logging.warning(
                'Bucket input queue is empty when calling next_batch. Bucket queue size: %i, Input queue size: %i',
                self._batch_queue.qsize(), self._example_queue.qsize())
            if self.single_pass and self._finished_reading:
                tf.compat.v1.logging.info("Finished reading dataset in single_pass mode.")
                return None
            return None

        batch = self._batch_queue.get()  # get the next Batch
        return batch

    def fill_example_queue(self):
        """
        加载数据,将m一行数据的source和target分开,构造Example实例,存入到_example_queue中
        """
        print('loading %s data...' % self._data_path)
        i = 0
        all_samples = []
        with open(self._data_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                all_samples.append(line.strip())
        if not self.is_test:
            random.shuffle(all_samples)
        for line in all_samples:
            splits = line.split('<SEP>')
            article = splits[0]
            abstract_sentences = [splits[1]]
            example = Example(article, abstract_sentences, self._vocab)
            self._example_queue.put(example)
        print('_example_queue size:%d' % self._example_queue.qsize())

    def fill_batch_queue(self):
        """
        初始化 _batch_queue,将数据按照bs分组后填入_batch_queue中.
        :return:
        """
        while not self._example_queue.empty():
            if self.mode == 'decode':
                # beam search decode mode single example repeated in the batch
                ex = self._example_queue.get()
                b = [ex for _ in range(self.batch_size)]
                self._batch_queue.put(Batch(b, self._vocab, self.batch_size))
            else:
                # Get bucketing_cache_size-many batches of Examples into a list, then sort
                inputs = []
                for _ in range(self.batch_size * self._bucketing_cache_size):
                    if self._example_queue.empty():
                        return
                    inputs.append(self._example_queue.get())
                inputs = sorted(inputs, key=lambda inp: inp.enc_len,
                                reverse=True)  # 按照encoder_len排序,为了优化训练过程

                # Group the sorted Examples into batches, optionally shuffle the batches, and place in the batch queue.
                batches = []
                for i in range(0, len(inputs), self.batch_size):    # 将所有数据按照batch_size分组
                    batches.append(inputs[i:i + self.batch_size])
                if not self.single_pass:
                    shuffle(batches)    # shuffle
                for b in batches:  # each b is a list of Example objects
                    self._batch_queue.put(Batch(b, self._vocab, self.batch_size))
