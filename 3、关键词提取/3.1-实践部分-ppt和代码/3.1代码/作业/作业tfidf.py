"""
下 载 新 浪 新 闻 门 户 公 开 数 据 集 ， 训 练 i d f ， 然 后 准 备 验 证 集 语 料 ，
对 比 tf i d f 和 textrank两种系统的效果；（有余力的同学可以自己标注一小部分语料，统计top k命 中率）
"""


import os
import time
import jieba
from keywords.tfidf import TFIDF


def get_stop_words():
    stop_words = set()
    with open('../stop_words.txt', 'r', encoding='utf-8') as f:
        for line in f:
            word = line.strip()
            if not word:
                continue
            stop_words.add(word)
    stop_words.add('\n')
    stop_words.add(' ')
    return stop_words


stop_words = get_stop_words()

DATA_DIR ='./data'
OUTPUT_PATH = './IDF.txt'
seg_files = [os.path.join(DATA_DIR, label) for label in os.listdir(DATA_DIR)]
tfidf_ext = TFIDF()
t1 = time.time()
# tfidf_ext.train_idf(seg_files=seg_files, output_file_name=OUTPUT_PATH) # 训练
print(f'train idf cost:{time.time()-t1}s')
print('test_tfidf~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
txt = "乌克兰释放参过军的囚犯参战"
words = [word for word in jieba.lcut(txt) if word not in stop_words]
print(words)
tfidf_ext.load_idf(OUTPUT_PATH)
result = tfidf_ext.compute_tfidf(words)
print(result)
