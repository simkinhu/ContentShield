import tensorflow as tf
import json
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

# 功能：加载保存的模型
def load_model(model_path):
    return tf.keras.models.load_model(model_path)

# 功能：加载 Tokenizer
def load_tokenizer(tokenizer_path):
    with open(tokenizer_path, 'r', encoding='utf-8') as f:
        tokenizer_json_string = f.read()  # 直接读取文件内容为字符串
    tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(tokenizer_json_string)
    return tokenizer


# 功能：处理新的文本数据
def prepare_text(texts, tokenizer, max_len=20):
    sequences = tokenizer.texts_to_sequences(texts)
    padded_sequences = pad_sequences(sequences, maxlen=max_len, padding='post', truncating='post')
    return padded_sequences

# 功能：进行预测
def predict(model, processed_data):
    predictions = model.predict(processed_data)
    predicted_class_indices = np.argmax(predictions, axis=1)
    predicted_labels = [label_dict[index] for index in predicted_class_indices]
    return predicted_labels

if __name__ == "__main__":
    date = '20240420'  # 设置日期
    train_dir = 'train'
    preprocess_dir = 'preprocess'

    model_path = f'{train_dir}/{date}/model.h5'
    tokenizer_path = f'{preprocess_dir}/{date}/tokenizer.json'

    # 定义标签映射直接在代码中
    label_dict = {
        0: "政治敏感",
        1: "违禁违规",
        2: "暴力极端",
        3: "色情内容",
        4: "侮辱性语言",
        5: "恐怖内容",
        6: "儿童不宜",
        7: "欺诈行为",
        8: "非法交易",
        9: "网络暴力",
        10: "自我伤害",
        11: "仇恨歧视",
        12: "不实信息",
        13: "性骚扰",
        14: "恶意推广",
        15: "其它"
    }

    # 加载模型和 Tokenizer
    model = load_model(model_path)
    tokenizer = load_tokenizer(tokenizer_path)

    # 测试文本数据
    test_texts = ["我国拟规定任何单位个人不得私自开拆他人邮件中新社北京四月二十二日", "我要回家吃饭", "你在哪里？", "哈哈哈哈"]

    # 准备数据
    processed_data = prepare_text(test_texts, tokenizer)

    # 进行预测
    predictions = predict(model, processed_data)

    # 打印预测结果
    for text, label in zip(test_texts, predictions):
        print(f"Text: {text}\nPredicted Label: {label}\n")
