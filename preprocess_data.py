import os
import numpy as np
import pandas as pd
from datetime import datetime
from transformers import BertTokenizer
from sklearn.model_selection import train_test_split
import json

class TextSafetyPreprocessor:
    def __init__(self, config):
        self.config = config
        self.labels_dict = self._load_labels_dict()
        self.tokenizer = BertTokenizer.from_pretrained(self.config['bert_model_name'])

    @staticmethod
    def _load_labels_dict():
        return {
            "政治敏感": 0, "违禁违规": 1, "暴力极端": 2, "色情内容": 3,
            "侮辱性语言": 4, "恐怖内容": 5, "儿童不宜": 6, "欺诈行为": 7,
            "非法交易": 8, "网络暴力": 9, "自我伤害": 10, "仇恨歧视": 11,
            "不实信息": 12, "性骚扰": 13, "恶意推广": 14, "其它": 15
        }

    def load_data(self):
        texts, labels = [], []
        for filename in os.listdir(self.config['data_dir']):
            if filename.endswith('.txt'):
                filepath = os.path.join(self.config['data_dir'], filename)
                with open(filepath, 'r', encoding='utf-8') as file:
                    for line in file:
                        parts = line.strip().split('\t', 1)
                        if len(parts) == 2:
                            label, text = parts
                            texts.append(text)
                            labels.append(self.labels_dict.get(label, self.labels_dict['其它']))
        return texts, labels

    def preprocess_texts(self, texts):
        encoded_texts = self.tokenizer(
            texts,
            padding='max_length',
            truncation=True,
            max_length=self.config['max_len'],
            return_tensors='np'
        )
        return encoded_texts

    def save_preprocessed_data(self, encoded_texts, labels):
        if not os.path.exists(self.config['output_dir']):
            os.makedirs(self.config['output_dir'])

        # Split data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            list(zip(encoded_texts['input_ids'], encoded_texts['attention_mask'])),
            labels,
            test_size=self.config['test_size'],
            random_state=42
        )

        # Unzip the tuples
        X_train_input_ids, X_train_attention_mask = zip(*X_train)
        X_test_input_ids, X_test_attention_mask = zip(*X_test)

        # Save train data
        np.save(os.path.join(self.config['output_dir'], 'X_train_input_ids.npy'), np.array(X_train_input_ids))
        np.save(os.path.join(self.config['output_dir'], 'X_train_attention_mask.npy'), np.array(X_train_attention_mask))
        np.save(os.path.join(self.config['output_dir'], 'y_train.npy'), np.array(y_train))

        # Save test data
        np.save(os.path.join(self.config['output_dir'], 'X_test_input_ids.npy'), np.array(X_test_input_ids))
        np.save(os.path.join(self.config['output_dir'], 'X_test_attention_mask.npy'), np.array(X_test_attention_mask))
        np.save(os.path.join(self.config['output_dir'], 'y_test.npy'), np.array(y_test))

        # Save labels dictionary
        with open(os.path.join(self.config['output_dir'], 'labels_dict.json'), 'w', encoding='utf-8') as f:
            json.dump(self.labels_dict, f, ensure_ascii=False, indent=4)

    def run(self):
        print("开始加载数据...")
        texts, labels = self.load_data()
        print(f"数据加载完成，共{len(texts)}条记录")

        print("开始预处理文本...")
        encoded_texts = self.preprocess_texts(texts)
        print("文本预处理完成")

        print("正在保存预处理后的数据...")
        self.save_preprocessed_data(encoded_texts, labels)
        print(f"预处理数据已保存到 {self.config['output_dir']}")

def main():
    config = {
        'data_dir': 'data',
        'output_dir': os.path.join('preprocess', datetime.now().strftime("%Y%m%d")),
        'bert_model_name': 'bert-base-chinese',
        'max_len': 128,
        'test_size': 0.2
    }

    preprocessor = TextSafetyPreprocessor(config)
    preprocessor.run()

if __name__ == "__main__":
    main()