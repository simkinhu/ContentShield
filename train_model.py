import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, GlobalAveragePooling1D, Dense
from tensorflow.keras.optimizers import Adam
import json
import os
from datetime import datetime


# 检测是否有GPU，如果有GPU则使用GPU，如果没有则使用CPU
def setup_device():
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # 设置TensorFlow只使用第一个GPU
            tf.config.set_visible_devices(gpus[0], 'GPU')
            tf.config.experimental.set_memory_growth(gpus[0], True)
            print("Using GPU")
        except RuntimeError as e:
            # 如果出现错误，就打印错误信息
            print(e)
    else:
        print("Using CPU")


# 功能：加载数据和标签
def load_data(data_path, labels_path):
    data = np.load(data_path)
    labels = np.load(labels_path)
    return data, labels


# 功能：加载 Tokenizer
def load_tokenizer(tokenizer_path):
    with open(tokenizer_path, 'r', encoding='utf-8') as f:
        json_string = f.read()
    tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(json_string)
    return tokenizer


# 功能：构建新模型
def build_model(vocab_size, embedding_dim=16, input_length=20, num_classes=15):
    model = Sequential([
        Embedding(vocab_size, embedding_dim, input_length=input_length),
        GlobalAveragePooling1D(),
        Dense(24, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])
    return model


# 功能：加载现有模型或创建新模型
def load_or_create_model(model_path, tokenizer):
    if os.path.exists(model_path):
        print("加载现有模型继续训练。")
        return tf.keras.models.load_model(model_path)
    else:
        print("未找到模型，创建新模型。")
        vocab_size = len(tokenizer.word_index) + 1  # 索引从1开始
        return build_model(vocab_size)


if __name__ == "__main__":
    setup_device()
    base_path = 'preprocess/20240420'
    train_dir = 'train'
    today = datetime.now().strftime("%Y%m%d")
    model_dir = os.path.join(train_dir, f'epoch_{today}')

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    model_path = os.path.join(model_dir, 'model.h5')
    data_path = os.path.join(base_path, 'data.npy')
    labels_path = os.path.join(base_path, 'labels.npy')
    tokenizer_path = os.path.join(base_path, 'tokenizer.json')

    data, labels = load_data(data_path, labels_path)
    tokenizer = load_tokenizer(tokenizer_path)

    model = load_or_create_model(model_path, tokenizer)

    # 训练模型并打印每步的loss和accuracy
    model.fit(data, labels, epochs=30, batch_size=32, verbose=2)

    model.save(os.path.join(model_dir, 'model.h5'))
    print(f"模型已保存至: {os.path.join(model_dir, 'model.h5')}")
