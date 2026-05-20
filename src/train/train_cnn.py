"""
train_cnn.py
功能：加载频谱图数据集
      使用最优超参数训练CNN模型
      保存模型与训练曲线
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['STHeiti', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from PIL import Image
from tqdm import tqdm

# 添加项目根目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from src.models.cnn_model import build_cnn

# ============ 全局配置 ============
SPEC_DIR = "data/spectrograms"
MODEL_SAVE_PATH = "outputs/models/best_cnn_model.h5"
IMG_SIZE = (128, 128)
RANDOM_STATE = 42
BATCH_SIZE = 32
EPOCHS = 100

# 最优超参数（由调优实验确定）
# dropout=0.2在调优中达到70.71%验证集准确率，优于dropout=0.3的65.66%
BEST_LR = 0.0001
BEST_CONV_BLOCKS = 2
BEST_DROPOUT_CONV = 0.2
BEST_DROPOUT_FC = 0.5

GENRES = ['blues', 'classical', 'country', 'disco', 'hiphop',
          'jazz', 'metal', 'pop', 'reggae', 'rock']


def load_spectrogram_dataset():
    """
    从data/spectrograms/目录加载所有频谱图
    返回：图像数组X 和 标签数组y
    """
    X = []
    y = []

    for label_idx, genre in enumerate(GENRES):
        genre_dir = os.path.join(SPEC_DIR, genre)
        img_files = [f for f in os.listdir(genre_dir)
                     if f.endswith('.png')]

        print(f"加载 {genre} 类别，共{len(img_files)}张图像...")

        for img_file in tqdm(img_files):
            img_path = os.path.join(genre_dir, img_file)
            try:
                # 读取图像并转为灰度图
                img = Image.open(img_path).convert('L')
                # 调整尺寸
                img = img.resize(IMG_SIZE)
                # 转为numpy数组并归一化
                img_array = np.array(img) / 255.0
                X.append(img_array)
                y.append(label_idx)
            except Exception as e:
                print(f"读取图像 {img_file} 失败：{e}")

    X = np.array(X)
    y = np.array(y)

    # 增加通道维度 (N, 128, 128) -> (N, 128, 128, 1)
    X = X[..., np.newaxis]

    print(f"\n数据集加载完成！")
    print(f"X.shape: {X.shape}")
    print(f"y.shape: {y.shape}")
    print(f"标签分布：{np.bincount(y)}")

    return X, y


def split_dataset(X, y):
    """
    按 8:1:1 比例划分训练集、验证集、测试集
    """
    # 先划分出测试集（10%）
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y,
        test_size=0.1,
        random_state=RANDOM_STATE,
        stratify=y
    )
    # 再从剩余数据中划分验证集（约11%，使总体为10%）
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val,
        test_size=0.11,
        random_state=RANDOM_STATE,
        stratify=y_train_val
    )

    print(f"训练集大小：{X_train.shape[0]}")
    print(f"验证集大小：{X_val.shape[0]}")
    print(f"测试集大小：{X_test.shape[0]}")

    return X_train, X_val, X_test, y_train, y_val, y_test


def get_data_augmentation():
    """
    定义数据增强管道
    对训练集图像进行随机变换以增加样本多样性
    """
    data_augmentation = tf.keras.Sequential([
        # 随机水平翻转
        tf.keras.layers.RandomFlip('horizontal'),
        # 随机旋转（±10%圆周角）
        tf.keras.layers.RandomRotation(0.1),
        # 随机宽度平移
        tf.keras.layers.RandomTranslation(0.1, 0.1),
        # 随机缩放
        tf.keras.layers.RandomZoom(0.1),
    ], name='data_augmentation')

    return data_augmentation


def build_dataset_pipeline(X_train, y_train,
                            X_val, y_val,
                            X_test, y_test):
    """
    构建tf.data数据管道，提升训练效率
    """
    # 训练集：加入数据增强
    train_ds = tf.data.Dataset.from_tensor_slices(
        (X_train, y_train)
    )
    train_ds = (train_ds
                .shuffle(buffer_size=1000,
                         seed=RANDOM_STATE)
                .batch(BATCH_SIZE)
                .prefetch(tf.data.AUTOTUNE))

    # 验证集：不进行数据增强
    val_ds = tf.data.Dataset.from_tensor_slices(
        (X_val, y_val)
    )
    val_ds = (val_ds
              .batch(BATCH_SIZE)
              .prefetch(tf.data.AUTOTUNE))

    # 测试集：不进行数据增强
    test_ds = tf.data.Dataset.from_tensor_slices(
        (X_test, y_test)
    )
    test_ds = (test_ds
               .batch(BATCH_SIZE)
               .prefetch(tf.data.AUTOTUNE))

    return train_ds, val_ds, test_ds


def get_callbacks():
    """
    定义训练回调函数
    仅保留ModelCheckpoint，保存验证集准确率最高的模型
    注意：不使用EarlyStopping和ReduceLROnPlateau，
          因为lr=0.0001的模型需要较多epoch才能收敛，
          过早触发会导致训练失败
    """
    callbacks = [
        # 保存验证集准确率最高的模型
        tf.keras.callbacks.ModelCheckpoint(
            filepath=MODEL_SAVE_PATH,
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
    ]
    return callbacks


def plot_training_history(history):
    """
    绘制训练过程曲线
    包含Loss曲线和Accuracy曲线
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    epochs_range = range(1, len(history.history['loss']) + 1)

    # Loss曲线
    axes[0].plot(epochs_range,
                 history.history['loss'],
                 'b-', label='训练Loss', linewidth=2)
    axes[0].plot(epochs_range,
                 history.history['val_loss'],
                 'r-', label='验证Loss', linewidth=2)
    axes[0].set_xlabel('训练轮次 (Epoch)', fontsize=13)
    axes[0].set_ylabel('Loss', fontsize=13)
    axes[0].set_title('训练与验证Loss曲线', fontsize=15)
    axes[0].legend(fontsize=12)
    axes[0].grid(True, alpha=0.3)

    # Accuracy曲线
    axes[1].plot(epochs_range,
                 history.history['accuracy'],
                 'b-', label='训练准确率', linewidth=2)
    axes[1].plot(epochs_range,
                 history.history['val_accuracy'],
                 'r-', label='验证准确率', linewidth=2)
    axes[1].set_xlabel('训练轮次 (Epoch)', fontsize=13)
    axes[1].set_ylabel('准确率', fontsize=13)
    axes[1].set_title('训练与验证准确率曲线', fontsize=15)
    axes[1].legend(fontsize=12)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('outputs/figures/cnn_training_curve.png',
                dpi=150, bbox_inches='tight')
    print("训练曲线已保存至 outputs/figures/cnn_training_curve.png")


def plot_confusion_matrix(y_true, y_pred):
    """
    绘制混淆矩阵热力图
    """
    from sklearn.metrics import confusion_matrix
    import seaborn as sns

    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(12, 10))
    sns.heatmap(cm,
                annot=True,
                fmt='d',
                cmap='Blues',
                xticklabels=GENRES,
                yticklabels=GENRES,
                linewidths=0.5)
    plt.xlabel('预测标签', fontsize=13)
    plt.ylabel('真实标签', fontsize=13)
    plt.title('CNN模型混淆矩阵', fontsize=15)
    plt.xticks(rotation=45, ha='right', fontsize=11)
    plt.yticks(rotation=0, fontsize=11)
    plt.tight_layout()
    plt.savefig('outputs/figures/confusion_matrix.png',
                dpi=150, bbox_inches='tight')
    print("混淆矩阵已保存至 outputs/figures/confusion_matrix.png")


if __name__ == '__main__':
    # Step1：加载数据集
    print("=" * 50)
    print("Step1：加载频谱图数据集")
    print("=" * 50)
    X, y = load_spectrogram_dataset()

    # Step2：划分数据集
    print("\n" + "=" * 50)
    print("Step2：划分训练集/验证集/测试集")
    print("=" * 50)
    (X_train, X_val, X_test,
     y_train, y_val, y_test) = split_dataset(X, y)

    # Step3：构建数据管道
    print("\n" + "=" * 50)
    print("Step3：构建tf.data数据管道")
    print("=" * 50)
    train_ds, val_ds, test_ds = build_dataset_pipeline(
        X_train, y_train, X_val, y_val, X_test, y_test
    )

    # Step4：构建CNN模型
    print("\n" + "=" * 50)
    print("Step4：构建CNN模型")
    print("=" * 50)
    model = build_cnn(
        input_shape=(128, 128, 1),
        num_classes=10,
        num_conv_blocks=BEST_CONV_BLOCKS,
        dropout_rate_conv=BEST_DROPOUT_CONV,
        dropout_rate_fc=BEST_DROPOUT_FC,
        learning_rate=BEST_LR
    )
    model.summary()

    # Step5：训练模型
    print("\n" + "=" * 50)
    print("Step5：开始训练CNN模型")
    print("=" * 50)
    os.makedirs('outputs/models', exist_ok=True)
    os.makedirs('outputs/figures', exist_ok=True)

    history = model.fit(
        train_ds,
        epochs=EPOCHS,
        validation_data=val_ds,
        callbacks=get_callbacks(),
        verbose=1
    )

    # Step6：绘制训练曲线
    print("\n" + "=" * 50)
    print("Step6：绘制训练曲线")
    print("=" * 50)
    plot_training_history(history)

    # Step7：在测试集上评估
    print("\n" + "=" * 50)
    print("Step7：测试集评估")
    print("=" * 50)
    best_model = tf.keras.models.load_model(MODEL_SAVE_PATH)
    test_loss, test_acc = best_model.evaluate(test_ds, verbose=0)
    print(f"测试集Loss：{test_loss:.4f}")
    print(f"测试集准确率：{test_acc:.4f}")

    # Step8：生成混淆矩阵
    print("\n" + "=" * 50)
    print("Step8：生成混淆矩阵")
    print("=" * 50)
    y_pred = np.argmax(best_model.predict(test_ds), axis=1)
    plot_confusion_matrix(y_test, y_pred)

    from sklearn.metrics import classification_report
    print("\n各类别详细分类报告：")
    print(classification_report(y_test, y_pred,
                                target_names=GENRES))
