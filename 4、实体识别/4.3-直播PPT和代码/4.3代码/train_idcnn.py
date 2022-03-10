import evaluate
from ner.models.idcnn.idcnn_predictor import IDCNNPredictor
from ner.models.idcnn.idcnn_trainer import IDCNNCRFTrainer


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


# 读取数据
train_texts, train_labels = read_conll('./data/weiboNER.conll.train')
dev_texts, dev_labels = read_conll('./data/weiboNER.conll.dev')
test_texts, test_labels = read_conll('./data/weiboNER.conll.test')

# 实例化trainer，设置参数，训练
trainer = IDCNNCRFTrainer(
    './tmp/idcnn', filters=64, hidden_num=256, embedding_size=100, dropout_rate=0.3, learning_rate=1e-3,
    load_last_ckpt=True  # 训练前，加载上次训练好的模型，继续训练
)
trainer.train(
    train_texts, train_labels, validate_texts=dev_texts, validate_labels=dev_labels,
    batch_size=20, epoch=5, max_len=256  # 数据量少，需要多增加一些epoch
)

# 实例化predictor，加载模型
predictor = IDCNNPredictor('./tmp/idcnn')
predict_labels = predictor.predict(test_texts, batch_size=20)

# 将结果输出为3列
out = open('tmp/dev_results.txt', 'w', encoding='utf-8')
for text, each_true_labels, each_predict_labels in zip(test_texts, test_labels, predict_labels):
    for char, true_label, predict_label in zip(text, each_true_labels, each_predict_labels):
        out.write('{}\t{}\t{}\n'.format(char, true_label, predict_label))
    out.write('\n')
out.close()

# 评估
evaluate.eval('tmp/dev_results.txt')
