import numpy as np
import torch
import json
import evaluate
from ner.models.bert_crf.bert_crf_predictor import BERTCRFPredictor
from ner.models.bert_crf.bert_crf_trainer import BertCRFTrainer

# 设置随机种子
seed = 0
torch.manual_seed(seed)  # torch cpu随机种子
torch.cuda.manual_seed_all(seed)  # torch gpu随机种子
np.random.seed(seed)  # numpy随机种子


def read_conll(file_path):
    with open(file_path, encoding='utf-8', errors='ignore') as f:
        texts, labels = [], []
        for example in f.read().split('\n\n'):  # 迭代每条样本
            example = example.strip()
            if not example:
                continue
            texts.append([]), labels.append([])
            for term in example.split('\n'):
                if len(term.split('\t')) != 2:  # 跳过不合法的行
                    continue
                char, label = term.split('\t')
                texts[-1].append(char), labels[-1].append(label)
        return texts, labels

def load_cluener(data_dir):
    data = []
    tags_ = []
    with open(data_dir, 'r', encoding='utf8') as f:
        for line in f:
            line = line.strip()
            line = json.loads(line)
            text = line.get('text', '')
            labels = line.get('label', {})
            tags = ['O'] * len(text)
            for label, info in labels.items():
                for value, indexes in info.items():
                    for index in indexes:
                        tags[index[0]: index[1]+1] = [f'B-{label}'] + [f'I-{label}']*(index[1]-index[0])
            data.append(list(map(lambda x: x, text)))
            tags_.append(tags)
    return data, tags_

# 读取数据
# train_texts, train_labels = read_conll('./data/weiboNER.conll.train')
# dev_texts, dev_labels = read_conll('./data/weiboNER.conll.dev')
# test_texts, test_labels = read_conll('./data/weiboNER.conll.test')

train_texts, train_labels = load_cluener('./data/cluener_public/train.json')
dev_texts, dev_labels = load_cluener('./data/cluener_public/dev.json')
# test_texts, test_labels = load_cluener('./data/cluener_public/test.json')
test_texts, test_labels = dev_texts, dev_labels

bak_fix = ''
pretrained_model = 'bert-base-chinese'
pretrained_model_dir = f'../../../models/{pretrained_model}'
# 实例化trainer，设置参数，训练
# trainer = BertCRFTrainer(
#     pretrained_model_dir=pretrained_model_dir, model_dir=f'./tmp/bercrf-{pretrained_model}-{bak_fix}', learning_rate=4e-5,
#     enable_parallel=True,  # # 启用并行训练
#     # # # 设置focal loss，设置label权重分布（当前16个label, O的label_id为1），降低O标签权重
#     loss_type='focal_loss', focal_loss_gamma=2, focal_loss_alpha=[1.] + [0.3] + [1.] * 14
# )
# trainer.train(
#     train_texts, train_labels, validate_texts=dev_texts, validate_labels=dev_labels, batch_size=32, epoch=5
# )

# 实例化predictor，加载模型
predictor = BERTCRFPredictor(
    pretrained_model_dir=pretrained_model_dir, model_dir=f'./tmp/bercrf-{pretrained_model}{bak_fix}',
    enable_parallel=True  # # 启用并行预测
)
predict_labels = predictor.predict(test_texts, batch_size=20)

# 将结果输出为3列
out = open('tmp/dev_results.txt', 'w', encoding='utf-8')
for text, each_true_labels, each_predict_labels in zip(test_texts, test_labels, predict_labels):
    for char, true_label, predict_label in zip(text, each_true_labels, each_predict_labels):
        out.write('{}\t{}\t{}\n'.format(char, true_label, predict_label))
    out.write('\n')
out.close()

# 评估
# evaluate.eval('tmp/dev_results.txt')
evaluate.eval('tmp/dev_results.txt', entity_types=['name', 'company', 'game', 'organization', 'movie', 'address', 'position', 'government', 'scene', 'book'])

