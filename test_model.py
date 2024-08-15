import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences
import json

class TextSafetyClassifier:
    def __init__(self, model_path, tokenizer_path, config_path, max_len=100):
        self.model = self.load_model(model_path)
        self.tokenizer = self.load_tokenizer(tokenizer_path)
        self.config = self.load_config(config_path)
        self.max_len = max_len
        self.label_dict = self.create_label_dict()

    @staticmethod
    def load_model(model_path):
        return tf.keras.models.load_model(model_path)

    @staticmethod
    def load_tokenizer(tokenizer_path):
        with open(tokenizer_path, 'r', encoding='utf-8') as f:
            return tokenizer_from_json(f.read())

    @staticmethod
    def load_config(config_path):
        with open(config_path, 'r') as f:
            return json.load(f)

    def create_label_dict(self):
        # 定义实际的类别名称
        categories = [
            "政治敏感", "违禁违规", "暴力极端", "色情内容",
            "侮辱性语言", "恐怖内容", "儿童不宜", "欺诈行为",
            "非法交易", "网络暴力", "自我伤害", "仇恨歧视",
            "不实信息", "性骚扰", "恶意推广", "其它"
        ]
        return {i: category for i, category in enumerate(categories)}

    def preprocess_text(self, text):
        sequences = self.tokenizer.texts_to_sequences([text])
        padded_sequences = pad_sequences(sequences, maxlen=self.max_len, padding='post', truncating='post')
        return padded_sequences

    def classify_text(self, text):
        preprocessed_text = self.preprocess_text(text)
        prediction = self.model.predict(preprocessed_text)
        predicted_class = np.argmax(prediction, axis=1)[0]
        confidence = prediction[0][predicted_class]
        return self.label_dict[predicted_class], confidence

def main():
    # 设置路径
    base_dir = 'train'
    model_dir = 'model_resumable'
    model_path = os.path.join(base_dir, model_dir, 'final_model.keras')
    tokenizer_path = os.path.join('preprocess', '20240815', 'tokenizer.json')
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
        print(f"置信度: {confidence:.2f}")
        print("-" * 50)

if __name__ == "__main__":
    main()