import torch.nn as nn
from transformers import AlbertTokenizer, AlbertModel
from transformers import BertTokenizer, BertModel
from transformers import ElectraTokenizer, ElectraModel
from transformers import AutoTokenizer, AutoModelForMaskedLM
from ner.models.crf.crf_layer import CRF


class BertCRFModel(nn.Module):

    def __init__(self, bert_base_model_dir, label_size, drop_out_rate=0.5):
        super(BertCRFModel, self).__init__()
        self.label_size = label_size

        if 'albert' in bert_base_model_dir.lower():
            # 注意albert base使用bert tokenizer，参考https://huggingface.co/voidful/albert_chinese_base
            self.bert_tokenizer = BertTokenizer.from_pretrained(bert_base_model_dir)
            self.bert_model = AlbertModel.from_pretrained(bert_base_model_dir)
            # self.bert_tokenizer = AutoTokenizer.from_pretrained("voidful/albert_chinese_base",
            #                                                     cache_dir='./model/albert_chinese_base')
            # #
            # self.bert_model = AutoModelForMaskedLM.from_pretrained("voidful/albert_chinese_base", 
            #                                                        cache_dir='./model/albert_chinese_base')
        elif 'electra' in bert_base_model_dir.lower():
            self.bert_tokenizer = ElectraTokenizer.from_pretrained(bert_base_model_dir)
            self.bert_model = ElectraModel.from_pretrained(bert_base_model_dir)
        else:
            self.bert_tokenizer = BertTokenizer.from_pretrained(bert_base_model_dir)
            self.bert_model = BertModel.from_pretrained(bert_base_model_dir)

        self.dropout = nn.Dropout(drop_out_rate)
        self.linear = nn.Linear(self.bert_model.config.hidden_size, label_size)

        self.crf = CRF(label_size)  # # 定义CRF层

    def forward(self, input_ids, attention_mask, token_type_ids=None, position_ids=None,
                head_mask=None, inputs_embeds=None, labels=None):
        bert_out = self.bert_model(
            input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,
            position_ids=position_ids, head_mask=head_mask, inputs_embeds=None
        )
        if isinstance(self.bert_model, ElectraModel):
            last_hidden_state, = bert_out
        else:
            last_hidden_state, pooled_output = bert_out
        seq_outs = self.dropout(last_hidden_state)
        logits = self.linear(seq_outs)

        lengths = attention_mask.sum(axis=1)
        best_paths = self.crf.get_batch_best_path(logits, lengths)

        if labels is not None:
            # # 将交叉熵损失换成crf_loss
            # 计算loss时，忽略[CLS]、[SEP]以及PAD部分
            loss = self.crf.negative_log_loss(inputs=logits[:, 1:, :], length=lengths - 2, tags=labels[:, 1:])
            return best_paths, loss

        return best_paths  # # 直接返回预测的labels

    def get_bert_tokenizer(self):
        return self.bert_tokenizer
