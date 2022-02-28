"""
先把 新浪数据集数据处理好，保存文件成文件

"""
import os
import jieba
import text_normalize

NUM = 2000  # 每个类别取多少数据
DATA_PATH = "E:\python_work\THUCNews"
SAVE_DIR = './data'


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
labels = os.listdir(DATA_PATH)

for label in labels:
    label_dir = os.path.join(DATA_PATH, label)
    files = os.listdir(label_dir)
    file_num = len(files)
    save_path = os.path.join(SAVE_DIR, f"{label}.txt")
    with open(save_path, encoding='utf8', mode='w') as f_t:
        for file in files[:min(file_num, NUM)]:
            file_path = os.path.join(label_dir, file)
            with open(file_path, encoding='utf8', mode='r') as f_s:
                data = text_normalize.string_q2b(f_s.read().strip())

                seg = [word for word in jieba.lcut(data) if word not in stop_words]
                f_t.write(' '.join(seg))
                f_t.write('\n')