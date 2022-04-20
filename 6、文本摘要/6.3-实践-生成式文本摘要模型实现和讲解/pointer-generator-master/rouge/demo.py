#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : humeng
# @Time    : 2021/1/15

# scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
from rouge.rouge import Rouge

rouge = Rouge()
scores = rouge.get_scores(['The quick brown fox jumps over the lazy dog','The quick brown fox jumps over the lazy dog'],
                      ['The quick brown dog jumps on the log.', 'The quick brown dog jumps on the log.'])
print(scores)