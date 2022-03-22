import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from transformers import AlbertModel
from transformers import BertTokenizer, BertModel
from transformers import DistilBertTokenizer, DistilBertModel
from transformers import ElectraTokenizer, ElectraModel
from transformers import LongformerModel
from transformers.activations import get_activation

from classification.focal_loss import FocalLoss
from classification.label_smoothing_loss import LabelSmoothingCrossEntropy


class BertFCModel(nn.Module):

    def __init__(self, bert_base_model_dir, label_size, drop_out_rate=0.5,
                 loss_type='cross_entropy_loss', focal_loss_gamma=2, focal_loss_alpha=None):
        super(BertFCModel, self).__init__()
        self.label_size = label_size

        assert loss_type in ('cross_entropy_loss', 'focal_loss', 'label_smoothing_loss')  # 支持label_smoothing_loss
        if focal_loss_alpha:  # 确保focal_loss_alpha合法，必须是一个label的概率分布
            assert isinstance(focal_loss_alpha, list) and len(focal_loss_alpha) == label_size
        self.loss_type = loss_type  # 添加loss_type
        self.focal_loss_gamma = focal_loss_gamma
        self.focal_loss_alpha = focal_loss_alpha

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 根据预训练文件名，自动检测bert的各种变体，并加载
        if 'albert' in bert_base_model_dir.lower():
            self.bert_tokenizer = BertTokenizer.from_pretrained(bert_base_model_dir)
            self.bert_model = AlbertModel.from_pretrained(bert_base_model_dir)
        elif 'electra' in bert_base_model_dir.lower():
            self.bert_tokenizer = ElectraTokenizer.from_pretrained(bert_base_model_dir)
            self.bert_model = ElectraModel.from_pretrained(bert_base_model_dir)
        elif 'longformer' in bert_base_model_dir.lower():  # # 自动加载longformer模型
            self.bert_tokenizer = BertTokenizer.from_pretrained(bert_base_model_dir)
            # # longformer-chinese-base-4096模型参数prefix为bert而非标准的longformer，这是个坑
            LongformerModel.base_model_prefix = 'bert'
            self.bert_model = LongformerModel.from_pretrained(bert_base_model_dir)
        elif 'distil' in bert_base_model_dir.lower():  # # # 加载distil-bert模型
            self.bert_tokenizer = DistilBertTokenizer.from_pretrained(bert_base_model_dir)
            self.bert_model = DistilBertModel.from_pretrained(bert_base_model_dir)
        else:
            self.bert_tokenizer = BertTokenizer.from_pretrained(bert_base_model_dir)
            self.bert_model = BertModel.from_pretrained(bert_base_model_dir)

        self.dropout = nn.Dropout(drop_out_rate)
        # 定义linear层，进行 hidden_size->hidden_size 的映射，可以使用其作为bert pooler
        # 原始bert pooler是一个使用tanh激活的全连接层
        self.linear = nn.Linear(self.bert_model.config.hidden_size, self.bert_model.config.hidden_size)
        # 全连接分类层
        self.cls_layer = nn.Linear(self.bert_model.config.hidden_size, label_size)

    def forward(self, input_ids, attention_mask, token_type_ids=None, position_ids=None,
                head_mask=None, inputs_embeds=None, labels=None, return_extra=False):  #### 增加return_extra参数

        # # Longformer的[cls]位置设置全局Attention，值为2
        if isinstance(self.bert_model, LongformerModel):
            attention_mask[:, 0] = 2
            bert_out = self.bert_model(
                input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,
                position_ids=position_ids, inputs_embeds=None, return_dict=False,
                output_hidden_states=True  #### 增加output_hidden_states, longformer的output_attentions会报错，暂时忽略
            ) + (None,)  #### 暂时将longformer的attentions设置为None
        elif isinstance(self.bert_model, DistilBertModel):
            # # # distilbert不支持token_type_ids、position_ids参数，不传入
            bert_out = self.bert_model(
                input_ids=input_ids, attention_mask=attention_mask, inputs_embeds=None, return_dict=False,
                output_hidden_states=True, output_attentions=True  #### 增加output_hidden_states、output_attentions
            )
        else:
            bert_out = self.bert_model(
                input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,
                position_ids=position_ids, inputs_embeds=None, return_dict=False,
                output_hidden_states=True, output_attentions=True  #### 增加output_hidden_states、output_attentions
            )
        bert_out, (hidden_states, attentions) = \
            bert_out[:-2], bert_out[-2:]  #### 截取hidden_states attentions，保持原始bert_out不变

        if isinstance(self.bert_model, ElectraModel):
            # Electra使用gelu激活，自己实现gelu的bert pooler
            last_hidden_state, = bert_out
            x = last_hidden_state[:, 0, :]  # take <s> token (equiv. to [CLS])
            x = self.dropout(x)
            x = self.linear(x)
            x = get_activation("gelu")(x)  # although BERT uses tanh here, it seems Electra authors used gelu here
            x = self.dropout(x)
            pooled_output = x
        elif isinstance(self.bert_model, DistilBertModel):
            # # # distilbert使用relu激活的bert pooler
            last_hidden_state, = bert_out
            x = last_hidden_state[:, 0, :]
            x = self.linear(x)
            x = get_activation("relu")(x)
            x = self.dropout(x)
            pooled_output = x
        else:
            last_hidden_state, pooled_output = bert_out[:2]
        cls_outs = self.dropout(pooled_output)
        logits = self.cls_layer(cls_outs)

        best_labels = torch.argmax(logits, dim=-1)
        best_labels = best_labels.to(self.device)

        #### 将logits hiddens attentions装进extra，蒸馏时使用
        extra = {'hiddens': hidden_states, 'logits': logits, 'attentions': attentions}

        if labels is not None:
            # 根据不同的loss_type，选择不同的loss计算
            if self.loss_type == 'cross_entropy_loss':
                loss = CrossEntropyLoss(ignore_index=-1)(logits, labels)  # label=-1被忽略
            elif self.loss_type == 'focal_loss':
                loss = FocalLoss(gamma=self.focal_loss_gamma, alpha=self.focal_loss_alpha)(logits, labels)
            else:
                loss = LabelSmoothingCrossEntropy(alpha=0.1)(logits, labels)  # 支持label_smoothing_loss
            return (best_labels, loss) if not return_extra else (best_labels, loss, extra)  #### 增加返回extra

        return best_labels if not return_extra else (best_labels, extra)  #### 增加返回extra

    def get_bert_tokenizer(self):
        return self.bert_tokenizer
