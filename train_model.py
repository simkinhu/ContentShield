import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Embedding, GlobalAveragePooling1D, Dense, Dropout, LSTM
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard, ReduceLROnPlateau
from tensorflow.keras.preprocessing.text import tokenizer_from_json
import json
from datetime import datetime

class TextSafetyTrainer:
    def __init__(self, config):
        self.config = config
        self.setup_gpu()
        self.load_data()
        self.load_tokenizer()
        self.load_or_build_model()

    def setup_gpu(self):
        print(f"TensorFlow版本: {tf.__version__}")
        print(f"CUDA是否可用: {tf.test.is_built_with_cuda()}")
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                tf.config.set_visible_devices(gpus[0], 'GPU')
                print(f"使用GPU进行训练: {gpus[0]}")
                print(f"GPU内存: {tf.config.experimental.get_memory_info('GPU:0')['current'] / (1024 * 1024)} MB")
            except RuntimeError as e:
                print(f"GPU设置错误: {e}")
        else:
            print("未检测到GPU，使用CPU进行训练")

    def load_data(self):
        self.X_train = np.load(os.path.join(self.config['data_dir'], 'X_train.npy'))
        self.y_train = np.load(os.path.join(self.config['data_dir'], 'y_train.npy'))
        self.X_test = np.load(os.path.join(self.config['data_dir'], 'X_test.npy'))
        self.y_test = np.load(os.path.join(self.config['data_dir'], 'y_test.npy'))
        print(f"数据加载完成。训练集大小: {self.X_train.shape[0]}, 测试集大小: {self.X_test.shape[0]}")

    def load_tokenizer(self):
        with open(os.path.join(self.config['data_dir'], 'tokenizer.json'), 'r', encoding='utf-8') as f:
            self.tokenizer = tokenizer_from_json(f.read())
        print("Tokenizer加载完成")

    def load_or_build_model(self):
        checkpoint_path = os.path.join(self.config['model_dir'], 'checkpoint.keras')
        if os.path.exists(checkpoint_path):
            print("找到已有的模型检查点，正在加载...")
            self.model = load_model(checkpoint_path)
            print("模型加载完成")
        else:
            print("未找到模型检查点，正在构建新模型...")
            self.build_model()

    def build_model(self):
        vocab_size = len(self.tokenizer.word_index) + 1
        self.model = Sequential([
            Embedding(vocab_size, self.config['embedding_dim'], input_length=self.config['max_length']),
            LSTM(128, return_sequences=True),
            GlobalAveragePooling1D(),
            Dense(256, activation='relu'),
            Dropout(0.5),
            Dense(128, activation='relu'),
            Dropout(0.3),
            Dense(self.config['num_classes'], activation='softmax')
        ])
        self.model.compile(
            loss='sparse_categorical_crossentropy',
            optimizer=Adam(learning_rate=self.config['learning_rate']),
            metrics=['accuracy']
        )
        print(self.model.summary())

    def train(self):
        checkpoint_path = os.path.join(self.config['model_dir'], 'checkpoint.keras')
        callbacks = [
            ModelCheckpoint(
                filepath=checkpoint_path,
                save_best_only=True,
                monitor='val_accuracy',
                save_weights_only=False
            ),
            EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=5,
                min_lr=0.0001
            ),
            TensorBoard(log_dir=os.path.join(self.config['model_dir'], 'logs'))
        ]

        # 获取初始epoch
        initial_epoch = 0
        if os.path.exists(checkpoint_path):
            training_state_path = os.path.join(self.config['model_dir'], 'training_state.json')
            if os.path.exists(training_state_path):
                with open(training_state_path, 'r') as f:
                    training_state = json.load(f)
                    initial_epoch = training_state['epoch']

        history = self.model.fit(
            self.X_train, self.y_train,
            epochs=self.config['epochs'],
            initial_epoch=initial_epoch,
            batch_size=self.config['batch_size'],
            validation_data=(self.X_test, self.y_test),
            callbacks=callbacks,
            verbose=1
        )

        # 保存训练状态
        with open(os.path.join(self.config['model_dir'], 'training_state.json'), 'w') as f:
            json.dump({'epoch': self.config['epochs']}, f)

        self.model.save(os.path.join(self.config['model_dir'], 'final_model.keras'))
        print(f"训练完成。最终模型保存在: {os.path.join(self.config['model_dir'], 'final_model.keras')}")

        test_loss, test_accuracy = self.model.evaluate(self.X_test, self.y_test, verbose=0)
        print(f"测试集准确率: {test_accuracy:.4f}")
        print(f"测试集损失: {test_loss:.4f}")

        return history

def main():
    config = {
        'data_dir': 'preprocess/20240815',
        'model_dir': os.path.join('train', 'model_resumable'),
        'embedding_dim': 200,
        'max_length': 100,
        'num_classes': 16,
        'learning_rate': 0.001,
        'epochs': 10,
        'batch_size': 128
    }

    if not os.path.exists(config['model_dir']):
        os.makedirs(config['model_dir'])

    with open(os.path.join(config['model_dir'], 'config.json'), 'w') as f:
        json.dump(config, f, indent=4)

    trainer = TextSafetyTrainer(config)
    history = trainer.train()

if __name__ == "__main__":
    main()