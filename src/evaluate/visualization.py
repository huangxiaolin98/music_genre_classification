"""
visualization.py
功能：结果可视化工具模块
      提供频谱图展示、混淆矩阵、模型对比图等绘图函数
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['STHeiti', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
import seaborn as sns
from sklearn.metrics import confusion_matrix

GENRES = ['blues', 'classical', 'country', 'disco', 'hiphop',
          'jazz', 'metal', 'pop', 'reggae', 'rock']


def plot_spectrogram_samples(spectrograms, labels, save_path=None):
    """
    展示各流派梅尔频谱图样本
    参数：
        spectrograms - 频谱图数组列表
        labels       - 对应标签列表
        save_path    - 保存路径（可选）
    """
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))

    for idx, (spec, label) in enumerate(
            zip(spectrograms, labels)):
        row = idx // 5
        col = idx % 5
        axes[row, col].imshow(spec, aspect='auto',
                              origin='lower', cmap='viridis')
        axes[row, col].set_title(GENRES[label], fontsize=13)
        axes[row, col].axis('off')

    plt.suptitle('各流派梅尔频谱图样本', fontsize=16)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"频谱图样本已保存至 {save_path}")


def plot_confusion_matrix(y_true, y_pred, title='混淆矩阵',
                           save_path=None):
    """
    绘制混淆矩阵热力图
    参数：
        y_true    - 真实标签
        y_pred    - 预测标签
        title     - 图表标题
        save_path - 保存路径（可选）
    """
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
    plt.title(title, fontsize=15)
    plt.xticks(rotation=45, ha='right', fontsize=11)
    plt.yticks(rotation=0, fontsize=11)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"混淆矩阵已保存至 {save_path}")


def plot_training_curves(history, save_path=None):
    """
    绘制训练过程曲线（Loss和Accuracy）
    参数：
        history   - Keras训练历史对象
        save_path - 保存路径（可选）
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    epochs_range = range(1, len(history.history['loss']) + 1)

    # Loss曲线
    axes[0].plot(epochs_range, history.history['loss'],
                 'b-', label='训练Loss', linewidth=2)
    axes[0].plot(epochs_range, history.history['val_loss'],
                 'r-', label='验证Loss', linewidth=2)
    axes[0].set_xlabel('训练轮次 (Epoch)', fontsize=13)
    axes[0].set_ylabel('Loss', fontsize=13)
    axes[0].set_title('训练与验证Loss曲线', fontsize=15)
    axes[0].legend(fontsize=12)
    axes[0].grid(True, alpha=0.3)

    # Accuracy曲线
    axes[1].plot(epochs_range, history.history['accuracy'],
                 'b-', label='训练准确率', linewidth=2)
    axes[1].plot(epochs_range, history.history['val_accuracy'],
                 'r-', label='验证准确率', linewidth=2)
    axes[1].set_xlabel('训练轮次 (Epoch)', fontsize=13)
    axes[1].set_ylabel('准确率', fontsize=13)
    axes[1].set_title('训练与验证准确率曲线', fontsize=15)
    axes[1].legend(fontsize=12)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"训练曲线已保存至 {save_path}")


def plot_feature_importance(model, feature_names, top_n=20,
                             save_path=None):
    """
    绘制决策树特征重要性图
    参数：
        model         - 训练好的决策树模型
        feature_names - 特征名称列表
        top_n         - 显示前N个重要特征
        save_path     - 保存路径（可选）
    """
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]

    plt.figure(figsize=(12, 8))
    plt.barh(range(top_n),
             importances[indices][::-1],
             color='steelblue', alpha=0.8)
    plt.yticks(range(top_n),
               [feature_names[i] for i in indices][::-1],
               fontsize=11)
    plt.xlabel('特征重要性', fontsize=13)
    plt.title(f'决策树Top-{top_n}重要特征', fontsize=15)
    plt.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"特征重要性图已保存至 {save_path}")
