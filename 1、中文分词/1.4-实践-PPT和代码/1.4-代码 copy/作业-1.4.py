import math
import json


class HMMTrainer(object):
    """HMM模型训练代码"""

    def train(self, texts, labels):
        assert len(texts) == len(labels)

        emit_p, start_p, trans_p = {}, {}, {}

        # 统计三个参数矩阵中每个元素的频数
        for text, text_labels in zip(texts, labels):
            assert len(text) == len(text_labels) != 0

            # 更新初始状态矩阵
            if text_labels[0] not in start_p:
                start_p[text_labels[0]] = 0
            start_p[text_labels[0]] += 1

            # 更新状态转移矩阵
            for i in range(len(text) - 1):
                if text_labels[i] not in trans_p:
                    trans_p[text_labels[i]] = {}
                if text_labels[i + 1] not in trans_p[text_labels[i]]:
                    trans_p[text_labels[i]][text_labels[i + 1]] = 0
                trans_p[text_labels[i]][text_labels[i + 1]] += 1

            # 更新发射矩阵
            for i in range(len(text)):
                if text_labels[i] not in emit_p:
                    emit_p[text_labels[i]] = {}
                if text[i] not in emit_p[text_labels[i]]:
                    emit_p[text_labels[i]][text[i]] = 0
                emit_p[text_labels[i]][text[i]] += 1

        # 根据频数矩阵，更新概率
        tmp_sum = sum(start_p.values())
        start_p = {label: math.log(freq / tmp_sum) for label, freq in start_p.items()}

        for label1 in trans_p:
            tmp_sum = sum(trans_p[label1].values())
            for label2 in trans_p[label1]:
                trans_p[label1][label2] = math.log(trans_p[label1][label2] / tmp_sum)

        for label in emit_p:
            tmp_sum = sum(emit_p[label].values())
            for char in emit_p[label]:
                emit_p[label][char] = math.log(emit_p[label][char] / tmp_sum)

        return start_p, trans_p, emit_p


if __name__ == '__main__':
    texts = ['我很开心', '一马当先', '哈哈']
    labels = ['SSBE', 'BMME', 'SS']
    start_p, trans_p, emit_p = HMMTrainer().train(texts, labels)
    print('start_p:', json.dumps(start_p, indent=4))
    print('\ntrans_p:', json.dumps(trans_p, indent=4))
    print('\nemit_p:', json.dumps(emit_p, ensure_ascii=False, indent=4))
