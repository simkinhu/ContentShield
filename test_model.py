import os
import numpy as np
import tensorflow as tf
from transformers import BertTokenizer, TFBertForSequenceClassification
import json

class TextSafetyClassifier:
    def __init__(self, model_path, tokenizer_path, config_path):
        self.model = self.load_model(model_path)
        self.tokenizer = self.load_tokenizer(tokenizer_path)
        self.config = self.load_config(config_path)
        self.label_dict = self.create_label_dict()

    @staticmethod
    def load_model(model_path):
        return TFBertForSequenceClassification.from_pretrained(model_path)

    @staticmethod
    def load_tokenizer(tokenizer_path):
        return BertTokenizer.from_pretrained(tokenizer_path)

    @staticmethod
    def load_config(config_path):
        with open(config_path, 'r') as f:
            return json.load(f)

    def create_label_dict(self):
        categories = [
            "政治敏感", "违禁违规", "暴力极端", "色情内容",
            "侮辱性语言", "恐怖内容", "儿童不宜", "欺诈行为",
            "非法交易", "网络暴力", "自我伤害", "仇恨歧视",
            "不实信息", "性骚扰", "恶意推广", "其它"
        ]
        return {i: category for i, category in enumerate(categories)}

    def preprocess_text(self, text):
        encoded = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.config['max_length'],
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='tf'
        )
        return encoded['input_ids'], encoded['attention_mask']

    def classify_text(self, text):
        input_ids, attention_mask = self.preprocess_text(text)
        outputs = self.model(input_ids, attention_mask=attention_mask)
        prediction = tf.nn.softmax(outputs.logits, axis=-1)
        predicted_class = tf.argmax(prediction, axis=-1).numpy()[0]
        confidence = prediction.numpy()[0][predicted_class]
        return self.label_dict[predicted_class], confidence

def main():
    # 设置路径
    base_dir = 'train'
    model_dir = 'model_transformer'
    model_path = os.path.join(base_dir, model_dir, 'final_model')
    tokenizer_path = 'bert-base-chinese'
    config_path = os.path.join(base_dir, model_dir, 'config.json')

    # 初始化分类器
    classifier = TextSafetyClassifier(model_path, tokenizer_path, config_path)

    # 测试循环
    while True:
        text = input("请输入要审核的文本（输入'quit'退出）: ")
        if text.lower() == 'quit':
            break

        category, confidence = classifier.classify_text(text)
        print(f"分类结果: {category}")
        print(f"置信度: {confidence:.4f}")
        print("-" * 50)

if __name__ == "__main__":
    main()