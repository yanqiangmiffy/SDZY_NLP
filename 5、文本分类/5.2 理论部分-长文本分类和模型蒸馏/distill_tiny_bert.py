import os
import re
from functools import partial

from datasets.arrow_dataset import Dataset
from sklearn.metrics import accuracy_score
from textbrewer import DistillationConfig, TrainingConfig, GeneralDistiller
from torch.utils.data.dataloader import DataLoader
from transformers import AdamW

from classification.bert_fc.bert_fc_predictor import BertFCPredictor
from classification.bert_fc.bert_fc_trainer import BertFCTrainer

"""
微调蒸馏tiny-bert
使用bert-base当老师
"""


def read_data(data_path):
    """
    读取原始数据，返回titles、labels
    """
    titles, labels = [], []
    with open(data_path, 'r', encoding='utf-8') as f:
        print('current file:', data_path)
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            _, _, label, title, _ = line.split('_!_')
            titles.append(list(title)), labels.append([label])
        print(data_path, 'finish')
    return titles, labels


# 读取数据
train_path, dev_path, test_path = \
    'data/toutiao_cat_data.train.txt', 'data/toutiao_cat_data.dev.txt', 'data/toutiao_cat_data.test.txt'
(train_texts, train_labels), (dev_texts, dev_labels), (test_texts, test_labels) = \
    read_data(train_path), read_data(dev_path), read_data(test_path)
print('数据读取完成')

enable_parallel = False  # 启用DataParallel此处会使蒸馏速度变慢，此处禁用
# 获取老师模型、启用return_extra
# 通过BertFCPredictor获取teacher model
teacher_predictor = BertFCPredictor(
    '../5.1代码/model/chinese-roberta-wwm-ext', '../5.1代码/tmp/bertfc', enable_parallel=enable_parallel
)
teacher_model = teacher_predictor.model
teacher_model.forward = partial(teacher_model.forward, return_extra=True)  # 启用return_extra
print('teacher模型加载成功，label mapping:', teacher_predictor.vocab.id2tag)

# 获取学生模型、启用return_extra
# 通过BertFCTrainer获取student model
pretrained_model, model_dir = './model/TinyBERT_4L_zh', './tmp/bertfc'
student_trainer = BertFCTrainer(pretrained_model, model_dir, enable_parallel=enable_parallel)
student_trainer.vocab.build_vocab(labels=train_labels, build_texts=False, with_build_in_tag_id=False)
student_trainer._build_model()
student_trainer.vocab.save_vocab('{}/{}'.format(student_trainer.model_dir, student_trainer.vocab_name))
student_trainer._save_config()
student_model = student_trainer.model
student_model.forward = partial(student_model.forward, return_extra=True)  # 启用return_extra
print('student模型加载成功，label mapping:', student_trainer.vocab.id2tag)  # 确保学生老师label mapping要一致

# 封装dataset和dataloader，适配textbrewer接口
train_input_ids, train_att_mask, train_label_ids = student_trainer._transform_batch(
    train_texts, train_labels, max_length=128
)
train_dataset = Dataset.from_dict(
    {'input_ids': train_input_ids, 'attention_mask': train_att_mask, 'labels': train_label_ids}
)
train_dataset.set_format(type='torch')  # 设置data type为torch的tensor
train_dataloader = DataLoader(train_dataset, batch_size=64)

# 蒸馏配置
distill_config = DistillationConfig(
    # 设置温度系数temperature, tiny-bert论文作者使用1表现最好，一般大于1比较好
    temperature=4,
    # 设置ground truth loss权重
    hard_label_weight=1,
    # 设置预测层蒸馏loss（即soft label损失）为交叉熵，并稍微放大其权重
    kd_loss_type='ce', kd_loss_weight=1.2,
    # 配置中间层蒸馏映射
    intermediate_matches=[
        # 配置hidden蒸馏映射、维度映射
        {'layer_T': 0, 'layer_S': 0, 'feature': 'hidden', 'loss': 'hidden_mse', 'weight': 1,
         'proj': ['linear', 312, 768]},  # embedding层输出
        {'layer_T': 3, 'layer_S': 1, 'feature': 'hidden', 'loss': 'hidden_mse', 'weight': 1,
         'proj': ['linear', 312, 768]},
        {'layer_T': 6, 'layer_S': 2, 'feature': 'hidden', 'loss': 'hidden_mse', 'weight': 1,
         'proj': ['linear', 312, 768]},
        {'layer_T': 9, 'layer_S': 3, 'feature': 'hidden', 'loss': 'hidden_mse', 'weight': 1,
         'proj': ['linear', 312, 768]},
        {'layer_T': 12, 'layer_S': 4, 'feature': 'hidden', 'loss': 'hidden_mse', 'weight': 1,
         'proj': ['linear', 312, 768]},
        # 配置attention矩阵蒸馏映射，注意layer序号从0开始
        {"layer_T": 2, "layer_S": 0, "feature": "attention", "loss": "attention_mse", "weight": 1},
        {"layer_T": 5, "layer_S": 1, "feature": "attention", "loss": "attention_mse", "weight": 1},
        {"layer_T": 8, "layer_S": 2, "feature": "attention", "loss": "attention_mse", "weight": 1},
        {"layer_T": 11, "layer_S": 3, "feature": "attention", "loss": "attention_mse", "weight": 1},
    ]
)

# 训练配置
epoch = 20  # 使用大一点的epoch
optimizer = AdamW(student_model.parameters(), lr=1e-4)  # 使用大一点的lr
train_config = TrainingConfig(
    output_dir=model_dir, log_dir='./log',
    data_parallel=enable_parallel, ckpt_frequency=1  # 一个epoch存1次模型
)


# 配置model中logits hiddens attentions losses的获取方法
def simple_adaptor(batch, model_outputs):
    return {
        'logits': model_outputs[-1]['logits'], 'hidden': model_outputs[-1]['hiddens'],
        'attention': model_outputs[-1]['attentions'], 'losses': model_outputs[1],
    }


# 蒸馏
distiller = GeneralDistiller(
    train_config=train_config, distill_config=distill_config,
    model_T=teacher_model, model_S=student_model,
    adaptor_T=simple_adaptor, adaptor_S=simple_adaptor
)
with distiller:
    print('开始蒸馏')
    distiller.train(optimizer, train_dataloader, num_epochs=epoch)
    print('蒸馏结束')

# 重命名文件名
newest_model_name = sorted(  # 根据textbrewer模型文件格式，查找最新的模型
    [f for f in os.listdir(model_dir) if 'pkl' in f], key=lambda f: int(re.findall('\\d+', f)[0]), reverse=True
)[0]
print('use model:', newest_model_name)
model_path, new_model_path = '{}/{}'.format(model_dir, newest_model_name), '{}/bert_model.bin'.format(model_dir)
os.system('cp {} {}'.format(model_path, new_model_path))

# 实例化predictor，加载模型
predictor = BertFCPredictor(
    pretrained_model_dir=pretrained_model, model_dir=model_dir, enable_parallel=True
)
predict_labels = predictor.predict(test_texts, batch_size=64)

# 评估
test_acc = accuracy_score(test_labels, predict_labels)
print('test acc:', test_acc)
