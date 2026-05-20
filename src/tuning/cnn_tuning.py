"""
cnn_tuning.py
功能：CNN超参数调优实验
      包含学习率调优、网络深度调优、Dropout率调优
      生成调优对比图表
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

# 添加项目根目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from src.models.cnn_model import build_cnn

# ============ 全局配置 ============
EPOCHS_TUNING = 50    # 调优实验训练轮次
BATCH_SIZE = 32
RANDOM_STATE = 42


def tune_learning_rate(train_ds, val_ds):
    """
    学习率调优实验
    对比四种学习率下的训练过程
    """
    learning_rates = [0.1, 0.01, 0.001, 0.0001]
    histories = {}

    print("\n开始学习率调优实验...")
    for lr in learning_rates:
        print(f"\n当前学习率：{lr}")
        tf.keras.backend.clear_session()

        model = build_cnn(
            num_conv_blocks=3,
            dropout_rate_conv=0.25,
            dropout_rate_fc=0.5,
            learning_rate=lr
        )

        history = model.fit(
            train_ds,
            epochs=EPOCHS_TUNING,
            validation_data=val_ds,
            verbose=1
        )

        histories[lr] = history
        final_val_acc = max(history.history['val_accuracy'])
        print(f"  最终验证准确率：{final_val_acc:.4f}")

    # 绘制不同学习率下的Loss曲线对比图
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    colors = ['red', 'blue', 'green', 'orange']
    labels = [f'lr={lr}' for lr in learning_rates]

    for idx, lr in enumerate(learning_rates):
        history = histories[lr]
        epochs_range = range(1,
                             len(history.history['loss']) + 1)
        # Loss曲线
        axes[0].plot(epochs_range,
                     history.history['loss'],
                     color=colors[idx],
                     label=labels[idx],
                     linewidth=2)
        # 验证准确率曲线
        axes[1].plot(epochs_range,
                     history.history['val_accuracy'],
                     color=colors[idx],
                     label=labels[idx],
                     linewidth=2)

    axes[0].set_xlabel('训练轮次 (Epoch)', fontsize=13)
    axes[0].set_ylabel('训练Loss', fontsize=13)
    axes[0].set_title('不同学习率下的Loss曲线对比', fontsize=15)
    axes[0].legend(fontsize=12)
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel('训练轮次 (Epoch)', fontsize=13)
    axes[1].set_ylabel('验证集准确率', fontsize=13)
    axes[1].set_title('不同学习率下的验证准确率对比', fontsize=15)
    axes[1].legend(fontsize=12)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('outputs/figures/lr_comparison.png',
                dpi=150, bbox_inches='tight')
    print("学习率调优图已保存至 outputs/figures/lr_comparison.png")

    # 输出汇总表
    print("\n学习率调优结果汇总：")
    print(f"{'学习率':<12}{'最终训练Loss':<18}"
          f"{'最终验证准确率':<18}{'收敛情况'}")
    print("-" * 65)
    for lr in learning_rates:
        h = histories[lr]
        final_loss = h.history['loss'][-1]
        final_val_acc = max(h.history['val_accuracy'])
        # 判断是否收敛（loss是否持续下降）
        loss_list = h.history['loss']
        converged = "是" if loss_list[-1] < loss_list[0] * 0.5 \
                    else "否"
        print(f"{lr:<12}{final_loss:<18.4f}"
              f"{final_val_acc:<18.4f}{converged}")

    # 找出最优学习率
    best_lr = max(learning_rates,
                  key=lambda lr: max(
                      histories[lr].history['val_accuracy']
                  ))
    print(f"\n最优学习率：{best_lr}")
    return best_lr


def tune_network_depth(train_ds, val_ds, best_lr):
    """
    网络深度调优实验
    对比2个、3个、4个卷积块的CNN性能差异
    """
    conv_blocks_list = [2, 3, 4]
    results = {}

    print("\n开始网络深度调优实验...")
    for num_blocks in conv_blocks_list:
        print(f"\n当前卷积块数量：{num_blocks}")
        tf.keras.backend.clear_session()

        model = build_cnn(
            num_conv_blocks=num_blocks,
            dropout_rate_conv=0.25,
            dropout_rate_fc=0.5,
            learning_rate=best_lr
        )

        # 计算参数量
        total_params = model.count_params()

        history = model.fit(
            train_ds,
            epochs=EPOCHS_TUNING,
            validation_data=val_ds,
            verbose=1
        )

        best_val_acc = max(history.history['val_accuracy'])
        results[num_blocks] = {
            'history': history,
            'best_val_acc': best_val_acc,
            'total_params': total_params
        }
        print(f"  参数量：{total_params:,}")
        print(f"  最优验证准确率：{best_val_acc:.4f}")

    # 绘制网络深度对比图
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    colors = ['steelblue', 'coral', 'green']

    for idx, num_blocks in enumerate(conv_blocks_list):
        history = results[num_blocks]['history']
        epochs_range = range(1,
                             len(history.history[
                                 'val_accuracy']) + 1)
        axes[0].plot(epochs_range,
                     history.history['val_accuracy'],
                     color=colors[idx],
                     label=f'{num_blocks}个卷积块',
                     linewidth=2)

    axes[0].set_xlabel('训练轮次 (Epoch)', fontsize=13)
    axes[0].set_ylabel('验证集准确率', fontsize=13)
    axes[0].set_title('不同网络深度验证准确率曲线对比',
                      fontsize=15)
    axes[0].legend(fontsize=12)
    axes[0].grid(True, alpha=0.3)

    # 最优验证准确率柱状图
    best_accs = [results[n]['best_val_acc']
                 for n in conv_blocks_list]
    param_counts = [results[n]['total_params'] / 1e6
                    for n in conv_blocks_list]
    x = range(len(conv_blocks_list))
    bars = axes[1].bar(x, best_accs,
                       color=colors,
                       alpha=0.8,
                       width=0.5)

    # 在柱子上方标注数值
    for bar, acc, params in zip(bars, best_accs, param_counts):
        axes[1].text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + 0.005,
                     f'{acc:.3f}\n({params:.1f}M参数)',
                     ha='center', va='bottom', fontsize=11)

    axes[1].set_xticks(x)
    axes[1].set_xticklabels(
        [f'{n}个卷积块' for n in conv_blocks_list],
        fontsize=12
    )
    axes[1].set_ylabel('最优验证准确率', fontsize=13)
    axes[1].set_title('不同网络深度最优性能对比', fontsize=15)
    axes[1].set_ylim(0, 1.05)
    axes[1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('outputs/figures/cnn_structure_tuning.png',
                dpi=150, bbox_inches='tight')
    print("网络深度调优图已保存至"
          " outputs/figures/cnn_structure_tuning.png")

    best_blocks = max(conv_blocks_list,
                      key=lambda n: results[n]['best_val_acc'])
    print(f"\n最优卷积块数量：{best_blocks}")
    return best_blocks


def tune_dropout_rate(train_ds, val_ds, best_lr, best_blocks):
    """
    Dropout率调优实验
    对比三种Dropout率下的训练/验证曲线
    观察过拟合程度变化
    """
    dropout_rates = [0.2, 0.3, 0.5]
    results = {}

    print("\n开始Dropout率调优实验...")
    for rate in dropout_rates:
        print(f"\n当前Dropout率：{rate}")
        tf.keras.backend.clear_session()

        model = build_cnn(
            num_conv_blocks=best_blocks,
            dropout_rate_conv=rate,
            dropout_rate_fc=rate * 2
            if rate * 2 <= 0.5 else 0.5,
            learning_rate=best_lr
        )

        history = model.fit(
            train_ds,
            epochs=EPOCHS_TUNING,
            validation_data=val_ds,
            verbose=1
        )

        final_train_acc = history.history['accuracy'][-1]
        best_val_acc = max(history.history['val_accuracy'])
        overfit_gap = final_train_acc - best_val_acc

        results[rate] = {
            'history': history,
            'final_train_acc': final_train_acc,
            'best_val_acc': best_val_acc,
            'overfit_gap': overfit_gap
        }
        print(f"  最终训练准确率：{final_train_acc:.4f}")
        print(f"  最优验证准确率：{best_val_acc:.4f}")
        print(f"  过拟合差距：{overfit_gap:.4f}")

    # 绘制Dropout调优对比图
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    colors = ['steelblue', 'coral', 'green']

    for idx, rate in enumerate(dropout_rates):
        history = results[rate]['history']
        epochs_range = range(1,
                             len(history.history[
                                 'accuracy']) + 1)
        # 训练准确率（实线）
        axes[0].plot(epochs_range,
                     history.history['accuracy'],
                     color=colors[idx],
                     linestyle='-',
                     label=f'训练 dropout={rate}',
                     linewidth=2)
        # 验证准确率（虚线）
        axes[0].plot(epochs_range,
                     history.history['val_accuracy'],
                     color=colors[idx],
                     linestyle='--',
                     label=f'验证 dropout={rate}',
                     linewidth=2)

    axes[0].set_xlabel('训练轮次 (Epoch)', fontsize=13)
    axes[0].set_ylabel('准确率', fontsize=13)
    axes[0].set_title('不同Dropout率下训练/验证准确率曲线',
                      fontsize=15)
    axes[0].legend(fontsize=10, ncol=2)
    axes[0].grid(True, alpha=0.3)

    # 过拟合差距对比柱状图
    gaps = [results[r]['overfit_gap'] for r in dropout_rates]
    val_accs = [results[r]['best_val_acc'] for r in dropout_rates]
    x = range(len(dropout_rates))

    ax2 = axes[1].twinx()
    bars = axes[1].bar(x, gaps,
                       color=colors,
                       alpha=0.6,
                       width=0.4,
                       label='过拟合差距（越小越好）')
    ax2.plot(x, val_accs, 'k-o',
             linewidth=2,
             markersize=8,
             label='最优验证准确率')

    axes[1].set_xticks(x)
    axes[1].set_xticklabels(
        [f'dropout={r}' for r in dropout_rates],
        fontsize=12
    )
    axes[1].set_ylabel('过拟合差距（训练-验证准确率）',
                       fontsize=12)
    ax2.set_ylabel('最优验证准确率', fontsize=12)
    axes[1].set_title('Dropout率对过拟合程度的影响', fontsize=15)
    axes[1].grid(True, alpha=0.3, axis='y')

    # 合并图例
    lines1, labels1 = axes[1].get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    axes[1].legend(lines1 + lines2,
                   labels1 + labels2,
                   fontsize=11,
                   loc='upper right')

    plt.tight_layout()
    plt.savefig('outputs/figures/dropout_tuning.png',
                dpi=150, bbox_inches='tight')
    print("Dropout调优图已保存至"
          " outputs/figures/dropout_tuning.png")

    best_rate = min(dropout_rates,
                    key=lambda r: results[r]['overfit_gap'])
    print(f"\n最优Dropout率：{best_rate}")
    return best_rate


if __name__ == '__main__':
    # 需要先运行 train_cnn.py 加载数据
    # 此处假设数据已加载完毕
    from src.train.train_cnn import (load_spectrogram_dataset,
                                      split_dataset,
                                      build_dataset_pipeline)

    print("=" * 50)
    print("加载数据集用于调优实验...")
    print("=" * 50)
    X, y = load_spectrogram_dataset()
    (X_train, X_val, X_test,
     y_train, y_val, y_test) = split_dataset(X, y)
    train_ds, val_ds, test_ds = build_dataset_pipeline(
        X_train, y_train,
        X_val, y_val,
        X_test, y_test
    )

    print("\n" + "=" * 50)
    print("实验一：学习率调优")
    print("=" * 50)
    best_lr = tune_learning_rate(train_ds, val_ds)

    print("\n" + "=" * 50)
    print("实验二：网络深度调优")
    print("=" * 50)
    best_blocks = tune_network_depth(train_ds, val_ds, best_lr)

    print("\n" + "=" * 50)
    print("实验三：Dropout率调优")
    print("=" * 50)
    best_rate = tune_dropout_rate(train_ds, val_ds,
                                   best_lr, best_blocks)

    print("\n" + "=" * 50)
    print("调优实验完成！最优超参数组合：")
    print("=" * 50)
    print(f"  最优学习率：{best_lr}")
    print(f"  最优卷积块数：{best_blocks}")
    print(f"  最优Dropout率：{best_rate}")
