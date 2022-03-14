import time

from ner.models.bert_crf.bert_crf_predictor_with_onnx import BERTCRFPredictor


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
test_texts, test_labels = read_conll('./data/weiboNER.conll.test')

# 实例化predictor，加载模型
predictor = BERTCRFPredictor(
    pretrained_model_dir='./model/bert-base-chinese', model_dir='./tmp/bercrf',
    enable_parallel=False  # 禁用并行预测
)

# 统计转换onnx前耗时
init_time = time.time()
predict_labels = predictor.predict(test_texts, batch_size=20, max_len=256)
print('cost_time: {:.4f}s'.format(time.time() - init_time))

# 统计onnx后耗时
predictor.transform2onnx(fix_seq_len=256)
init_time = time.time()
labels = predictor.predict(test_texts, batch_size=20, max_len=256)
print('cost_time: {:.4f}s'.format(time.time() - init_time))
