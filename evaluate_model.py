import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns

class TextSafetyEvaluator:
    def __init__(self, config):
        self.config = config
        self.model = self.load_model()
        self.data, self.labels = self.load_data()
        self.label_dict = self.load_label_dict()

    def load_model(self):
        print(f"正在加载模型: {self.config['model_path']}")
        return tf.keras.models.load_model(self.config['model_path'])

    def load_data(self):
        print(f"正在加载数据: {self.config['data_path']}, {self.config['labels_path']}")
        data = np.load(self.config['data_path'])
        labels = np.load(self.config['labels_path'])
        return data, labels

    def load_label_dict(self):
        print(f"正在加载标签字典: {self.config['label_dict_path']}")
        with open(self.config['label_dict_path'], 'r', encoding='utf-8') as file:
            label_dict = json.load(file)
        return {int(key): value for key, value in label_dict.items()}

    def evaluate_model(self):
        print("正在评估模型...")
        loss, accuracy = self.model.evaluate(self.data, self.labels, verbose=0)
        print(f"模型准确率: {accuracy*100:.2f}%")
        print(f"模型损失: {loss:.4f}")
        return self.model.predict(self.data)

    def generate_classification_report(self, true_classes, predicted_classes):
        target_names = [self.label_dict[idx] for idx in sorted(self.label_dict)]
        return classification_report(true_classes, predicted_classes, target_names=target_names)

    def plot_confusion_matrix(self, cm, class_names):
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(os.path.join(self.config['output_dir'], 'confusion_matrix.png'))
        print(f"混淆矩阵图已保存至: {os.path.join(self.config['output_dir'], 'confusion_matrix.png')}")

    def run_evaluation(self):
        predictions = self.evaluate_model()
        predicted_classes = np.argmax(predictions, axis=1)
        true_classes = np.argmax(self.labels, axis=1)

        cm = confusion_matrix(true_classes, predicted_classes)
        print("混淆矩阵:")
        print(cm)

        cr = self.generate_classification_report(true_classes, predicted_classes)
        print("分类报告:")
        print(cr)

        # 保存分类报告
        with open(os.path.join(self.config['output_dir'], 'classification_report.txt'), 'w') as f:
            f.write(cr)

        # 绘制并保存混淆矩阵
        class_names = [self.label_dict[i] for i in range(len(self.label_dict))]
        self.plot_confusion_matrix(cm, class_names)

def main():
    config = {
        'model_path': 'train/20240420/final_model.h5',  # 更新为最新的模型路径
        'data_path': 'preprocess/20240420/X_test.npy',  # 使用测试集数据
        'labels_path': 'preprocess/20240420/y_test.npy',  # 使用测试集标签
        'label_dict_path': 'preprocess/20240420/labels_dict.json',
        'output_dir': 'evaluation_results'
    }

    if not os.path.exists(config['output_dir']):
        os.makedirs(config['output_dir'])

    evaluator = TextSafetyEvaluator(config)
    evaluator.run_evaluation()

if __name__ == "__main__":
    main()