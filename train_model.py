import os
import numpy as np
import tensorflow as tf
from transformers import TFAutoModel, AutoTokenizer, TFAutoModelForSequenceClassification
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.utils.class_weight import compute_class_weight


class TextSafetyClassifier:
    def __init__(self, config):
        self.config = config
        self.setup_gpu()
        self.load_data()
        self.load_tokenizer()
        self.build_model()

    def setup_gpu(self):
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                tf.config.set_visible_devices(gpus[0], 'GPU')
                print(f"使用GPU进行训练: {gpus[0]}")
            except RuntimeError as e:
                print(f"GPU设置错误: {e}")
        else:
            print("未检测到GPU，使用CPU进行训练")

    def load_data(self):
        self.X_train = np.load(os.path.join(self.config['data_dir'], 'X_train.npy'), allow_pickle=True)
        self.y_train = np.load(os.path.join(self.config['data_dir'], 'y_train.npy'))

        # 确保 X_train 是字符串列表
        if isinstance(self.X_train[0], np.ndarray):
            self.X_train = [' '.join(map(str, x)) for x in self.X_train]
        elif not isinstance(self.X_train[0], str):
            raise ValueError("X_train 中的元素既不是字符串也不是数组")

        print(f"训练数据加载完成。训练集大小: {len(self.X_train)}")
        print(f"X_train 的第一个元素类型: {type(self.X_train[0])}")
        print(f"X_train 的第一个元素: {self.X_train[0][:100]}...")  # 打印前100个字符
        self.analyze_data_distribution()

    def analyze_data_distribution(self):
        all_classes = np.array(range(self.config['num_classes']))
        unique, counts = np.unique(self.y_train, return_counts=True)

        print("训练数据分布:")
        for label in all_classes:
            count = counts[unique == label][0] if label in unique else 0
            print(f"类别 {label}: {count} 样本")

        present_classes = np.unique(self.y_train)
        class_weights = compute_class_weight('balanced', classes=present_classes, y=self.y_train)
        self.class_weight_dict = dict(zip(present_classes, class_weights))

        for label in all_classes:
            if label not in self.class_weight_dict:
                self.class_weight_dict[label] = 1.0

        print("类别权重:", self.class_weight_dict)

    def load_tokenizer(self):
        tokenizer_path = self.config['basic_model_dir']
        print(f"尝试加载tokenizer的完整路径: {tokenizer_path}")
        if not os.path.exists(os.path.join(tokenizer_path, 'vocab.txt')):
            raise ValueError(f"Tokenizer vocab.txt not found at {tokenizer_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        print("Tokenizer从本地加载完成")

    def build_model(self):
        model_path = self.config['basic_model_dir']
        print(f"尝试加载模型的完整路径: {model_path}")
        if not os.path.exists(os.path.join(model_path, 'tf_model.h5')):
            raise ValueError(f"Model tf_model.h5 not found at {model_path}")

        base_model = TFAutoModelForSequenceClassification.from_pretrained(
            model_path,
            num_labels=self.config['num_classes'],
            from_pt=True
        )
        print("基础模型从本地加载完成")

        input_ids = tf.keras.layers.Input(shape=(self.config['max_length'],), dtype=tf.int32, name="input_ids")
        attention_mask = tf.keras.layers.Input(shape=(self.config['max_length'],), dtype=tf.int32,
                                               name="attention_mask")

        outputs = base_model(input_ids, attention_mask=attention_mask)[0]
        outputs = Dropout(0.1)(outputs)
        outputs = Dense(self.config['num_classes'], activation='softmax')(outputs)

        self.model = tf.keras.Model(inputs=[input_ids, attention_mask], outputs=outputs)

        self.model.compile(
            optimizer=Adam(learning_rate=self.config['learning_rate']),
            loss=SparseCategoricalCrossentropy(from_logits=False),
            metrics=['accuracy']
        )
        print(self.model.summary())

    def preprocess_data(self, texts):
        # 添加类型检查和转换
        if not isinstance(texts, list):
            raise ValueError("texts must be a list of strings")
        if not all(isinstance(text, str) for text in texts):
            raise ValueError("All elements in texts must be strings")

        return self.tokenizer(
            texts,
            padding='max_length',
            truncation=True,
            max_length=self.config['max_length'],
            return_tensors='tf'
        )

    def train(self):
        preprocessed_data = self.preprocess_data(self.X_train)

        epochs_dir = os.path.join(self.config['model_dir'], 'epochs')
        if not os.path.exists(epochs_dir):
            os.makedirs(epochs_dir)

        # 检查是否存在之前的checkpoint
        checkpoints = [f for f in os.listdir(epochs_dir) if f.endswith('.h5')]
        if checkpoints:
            latest_checkpoint = max(checkpoints, key=lambda x: int(x.split('-')[1].split('.')[0]))
            checkpoint_path = os.path.join(epochs_dir, latest_checkpoint)
            initial_epoch = int(latest_checkpoint.split('-')[1].split('.')[0])
            print(f"找到之前的checkpoint: {checkpoint_path}")
            self.model.load_weights(checkpoint_path)
            print(f"从epoch {initial_epoch} 继续训练")
        else:
            initial_epoch = 0
            print("未找到之前的checkpoint，从头开始训练")

        checkpoint_path = os.path.join(epochs_dir, 'epoch-{epoch:02d}.h5')
        callbacks = [
            ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True, save_best_only=False, save_freq='epoch'),
            EarlyStopping(monitor='loss', patience=3, restore_best_weights=True),
            ReduceLROnPlateau(monitor='loss', factor=0.2, patience=2, min_lr=0.00001)
        ]

        history = self.model.fit(
            {
                'input_ids': preprocessed_data['input_ids'],
                'attention_mask': preprocessed_data['attention_mask']
            },
            self.y_train,
            epochs=self.config['epochs'],
            initial_epoch=initial_epoch,
            batch_size=self.config['batch_size'],
            callbacks=callbacks,
            class_weight=self.class_weight_dict,
            verbose=1
        )

        self.model.save(os.path.join(self.config['model_dir'], 'final_model.h5'))
        print(f"训练完成。最终模型保存在: {os.path.join(self.config['model_dir'], 'final_model.h5')}")

        return history


def main():
    config = {
        'data_dir': 'preprocess/20240815',
        'model_dir': os.path.join('train', 'model_transformer'),
        'basic_model_dir': 'bert-base-chinese',
        'max_length': 128,
        'num_classes': 16,
        'learning_rate': 2e-5,
        'epochs': 10,
        'batch_size': 32
    }

    if not os.path.exists(config['model_dir']):
        os.makedirs(config['model_dir'])

    classifier = TextSafetyClassifier(config)
    classifier.train()


if __name__ == "__main__":
    main()