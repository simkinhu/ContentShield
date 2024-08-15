import numpy as np
import tensorflow as tf
from transformers import TFBertForSequenceClassification, BertTokenizer
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns


class TextSafetyEvaluator:
    def __init__(self, config):
        self.config = config
        self.model = self.load_model()
        self.tokenizer = self.load_tokenizer()
        self.data, self.labels = self.load_data()
        self.label_dict = self.load_label_dict()

    def load_model(self):
        print(f"正在加载模型: {self.config['model_path']}")
        return TFBertForSequenceClassification.from_pretrained(self.config['model_path'])

    def load_tokenizer(self):
        print(f"正在加载tokenizer: {self.config['tokenizer_path']}")
        return BertTokenizer.from_pretrained(self.config['tokenizer_path'])

    def load_data(self):
        print(
            f"正在加载数据: {self.config['input_ids_path']}, {self.config['attention_mask_path']}, {self.config['labels_path']}")
        input_ids = np.load(self.config['input_ids_path'])
        attention_mask = np.load(self.config['attention_mask_path'])
        labels = np.load(self.config['labels_path'])
        return {'input_ids': input_ids, 'attention_mask': attention_mask}, labels

    def load_label_dict(self):
        print(f"正在加载标签字典: {self.config['label_dict_path']}")
        with open(self.config['label_dict_path'], 'r', encoding='utf-8') as file:
            label_dict = json.load(file)
        return {int(key): value for key, value in label_dict.items()}

    def evaluate_model(self):
        print("正在评估模型...")
        predictions = self.model.predict(self.data)
        predicted_classes = np.argmax(predictions.logits, axis=1)
        true_classes = self.labels

        accuracy = np.mean(predicted_classes == true_classes)
        print(f"模型准确率: {accuracy * 100:.2f}%")

        return predictions.logits, true_classes

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
        plt.close()

    def plot_radar_chart(self, true_classes, predicted_classes):
        print("正在生成雷达图...")
        labels = list(self.label_dict.values())
        metrics = precision_recall_fscore_support(true_classes, predicted_classes, average=None)

        stats = ['Precision', 'Recall', 'F1-score']
        angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False)

        angles = np.concatenate((angles, [angles[0]]))

        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))

        for i, stat in enumerate(stats):
            values = list(metrics[i])
            values += values[:1]
            ax.plot(angles, values, 'o-', linewidth=2, label=stat)
            ax.fill(angles, values, alpha=0.25)

        ax.set_thetagrids(angles[:-1] * 180 / np.pi, labels)
        ax.set_ylim(0, 1)
        plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        plt.title('Performance Metrics by Category')

        plt.tight_layout()
        radar_chart_path = os.path.join(self.config['output_dir'], 'radar_chart.png')
        plt.savefig(radar_chart_path)
        print(f"雷达图已保存至: {radar_chart_path}")
        plt.close()

    def run_evaluation(self):
        predictions, true_classes = self.evaluate_model()
        predicted_classes = np.argmax(predictions, axis=1)

        cm = confusion_matrix(true_classes, predicted_classes)
        print("混淆矩阵:")
        print(cm)

        cr = self.generate_classification_report(true_classes, predicted_classes)
        print("分类报告:")
        print(cr)

        with open(os.path.join(self.config['output_dir'], 'classification_report.txt'), 'w') as f:
            f.write(cr)

        class_names = [self.label_dict[i] for i in range(len(self.label_dict))]
        self.plot_confusion_matrix(cm, class_names)

        self.plot_radar_chart(true_classes, predicted_classes)


def main():
    config = {
        'model_path': 'train/model_transformer/final_model',
        'tokenizer_path': 'bert-base-chinese',
        'input_ids_path': 'preprocess/20240815/X_test_input_ids.npy',
        'attention_mask_path': 'preprocess/20240815/X_test_attention_mask.npy',
        'labels_path': 'preprocess/20240815/y_test.npy',
        'label_dict_path': 'preprocess/20240815/labels_dict.json',
        'output_dir': 'evaluation_results'
    }

    if not os.path.exists(config['output_dir']):
        os.makedirs(config['output_dir'])

    evaluator = TextSafetyEvaluator(config)
    evaluator.run_evaluation()


if __name__ == "__main__":
    main()