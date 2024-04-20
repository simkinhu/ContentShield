import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
import json
import os

# 功能：加载保存的模型
def load_model(model_path):
    return tf.keras.models.load_model(model_path)

# 功能：加载数据和标签
def load_data(data_path, labels_path):
    data = np.load(data_path)
    labels = np.load(labels_path)
    return data, labels

# 功能：加载标签字典
def load_label_dict(label_dict_path):
    with open(label_dict_path, 'r') as file:
        label_dict = json.load(file)
    return {int(key): value for key, value in label_dict.items()}

# 功能：评估模型
def evaluate_model(model, data, labels):
    loss, accuracy = model.evaluate(data, labels, verbose=2)
    print(f"Model accuracy: {accuracy*100:.2f}%")
    print(f"Model loss: {loss:.4f}")
    return model.predict(data)

if __name__ == "__main__":
    date = '20230401'  # 设置日期
    base_dir = 'preprocess'
    train_dir = 'train'
    config_dir = 'config'

    # 路径设置
    model_path = os.path.join(train_dir, date, 'model.h5')
    data_path = os.path.join(base_dir, date, 'data.npy')
    labels_path = os.path.join(base_dir, date, 'labels.npy')
    label_dict_path = os.path.join(config_dir, 'labels.json')

    # 加载模型和数据
    model = load_model(model_path)
    data, labels = load_data(data_path, labels_path)
    label_dict = load_label_dict(label_dict_path)

    # 进行模型评估
    predictions = evaluate_model(model, data, labels)
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = np.argmax(labels, axis=1)

    # 生成混淆矩阵和分类报告
    cm = confusion_matrix(true_classes, predicted_classes)
    print("Confusion Matrix:")
    print(cm)

    # 生成分类报告
    target_names = [label_dict[idx] for idx in sorted(label_dict)]
    cr = classification_report(true_classes, predicted_classes, target_names=target_names)
    print("Classification Report:")
    print(cr)
