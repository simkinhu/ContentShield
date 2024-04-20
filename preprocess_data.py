import os
import numpy as np
from datetime import datetime
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# 定义标签映射直接在代码中
labels_dict = {
    "政治敏感": 0,
    "违禁违规": 1,
    "暴力极端": 2,
    "色情内容": 3,
    "侮辱性语言": 4,
    "恐怖内容": 5,
    "儿童不宜": 6,
    "欺诈行为": 7,
    "非法交易": 8,
    "网络暴力": 9,
    "自我伤害": 10,
    "仇恨歧视": 11,
    "不实信息": 12,
    "性骚扰": 13,
    "恶意推广": 14,
    "其它": 15
}

# 读取数据和标签
def load_data(directory):
    texts = []
    labels = []
    for filename in os.listdir(directory):
        if filename.endswith('.txt'):
            with open(os.path.join(directory, filename), 'r', encoding='utf-8') as file:
                for line in file:
                    label, text = line.strip().split(maxsplit=1)
                    texts.append(text)
                    labels.append(labels_dict[label])
    return texts, labels

# 文本预处理
def preprocess_texts(texts, num_words=10000, max_len=20):
    tokenizer = Tokenizer(num_words=num_words, oov_token="<OOV>")
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    padded_sequences = pad_sequences(sequences, maxlen=max_len, padding='post', truncating='post')
    return padded_sequences, tokenizer

# 保存预处理数据
def save_preprocessed_data(data, labels, save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    np.save(os.path.join(save_dir, 'data.npy'), data)
    np.save(os.path.join(save_dir, 'labels.npy'), labels)

if __name__ == "__main__":
    texts, labels = load_data('data')
    padded_sequences, tokenizer = preprocess_texts(texts)

    # 生成今天日期的文件夹，例如 20230401
    today = datetime.now().strftime("%Y%m%d")
    save_path = os.path.join('preprocess', today)
    save_preprocessed_data(padded_sequences, labels, save_path)

    # Optionally, save the tokenizer for further usage in training
    tokenizer_json = tokenizer.to_json()
    with open(os.path.join(save_path, 'tokenizer.json'), 'w', encoding='utf-8') as f:
        f.write(tokenizer_json)

    print("Preprocessing completed and data saved.")
