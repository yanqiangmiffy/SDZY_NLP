import unittest

from classification.bert_fc.bert_fc_predictor import BertFCPredictor
from classification.bert_fc.bert_fc_trainer import BertFCTrainer


class MyTestCase(unittest.TestCase):

    def setUp(self) -> None:
        self.model_dir = './tests/test_data'
        self.pretrained_model_dir = './model/longformer-chinese-base-4096'

    def test_trainer(self):
        texts = [
            ['天', '气', '真', '好'] * 200,  # 800字
            ['今', '天', '运', '气', '很', '差'] * 100,  # 600字
        ]
        labels = [
            ['正面'],
            ['负面']
        ]
        trainer = BertFCTrainer(self.pretrained_model_dir, self.model_dir, learning_rate=5e-5)
        trainer.train(texts, labels, validate_texts=texts, validate_labels=labels, batch_size=1, epoch=20)

    def test_predictor(self):
        predictor = BertFCPredictor(self.pretrained_model_dir, self.model_dir)
        texts = [
            ['天', '气', '真', '好'] * 200,
            ['今', '天', '运', '气', '很', '差'] * 100,
        ]
        labels = predictor.predict(texts)
        print(labels)


if __name__ == '__main__':
    unittest.main()
