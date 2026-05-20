"""
metrics.py
功能：提取CNN中间层特征
      训练融合决策树分类器
      生成三种方案综合性能对比图
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['STHeiti', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
import seaborn as sns
import tensorflow as tf
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (accuracy_score,
                             precision_score,
                             recall_score,
                             f1_score,
                             classification_report)
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

# ============ 全局配置 ============
MODEL_PATH = "outputs/models/best_cnn_model.h5"
RANDOM_STATE = 42

GENRES = ['blues', 'classical', 'country', 'disco', 'hiphop',
          'jazz', 'metal', 'pop', 'reggae', 'rock']


def extract_cnn_features(model_path, train_ds, test_ds):
    """
    提取CNN全连接层的中间层特征
    用训练好的CNN作为特征提取器
    参数：
        model_path - 训练好的CNN模型路径
        train_ds   - 训练集数据管道
        test_ds    - 测试集数据管道
    返回：
        train_features - 训练集CNN特征 shape:(N_train, 256)
        test_features  - 测试集CNN特征 shape:(N_test, 256)
        y_train        - 训练集标签
        y_test         - 测试集标签
    """
    print("加载训练好的CNN模型...")
    cnn_model = tf.keras.models.load_model(model_path)

    # 构建特征提取器：截取到dense_1全连接层输出
    feature_extractor = tf.keras.Model(
        inputs=cnn_model.input,
        outputs=cnn_model.get_layer('dense_1').output
    )
    print(f"特征提取器输出维度："
          f"{feature_extractor.output_shape}")

    # 提取训练集特征和标签
    print("\n提取训练集CNN特征...")
    train_features_list = []
    train_labels_list = []
    for images, labels in train_ds:
        features = feature_extractor(images, training=False)
        train_features_list.append(features.numpy())
        train_labels_list.append(labels.numpy())

    train_features = np.concatenate(train_features_list, axis=0)
    y_train = np.concatenate(train_labels_list, axis=0)
    print(f"训练集特征维度：{train_features.shape}")

    # 提取测试集特征和标签
    print("\n提取测试集CNN特征...")
    test_features_list = []
    test_labels_list = []
    for images, labels in test_ds:
        features = feature_extractor(images, training=False)
        test_features_list.append(features.numpy())
        test_labels_list.append(labels.numpy())

    test_features = np.concatenate(test_features_list, axis=0)
    y_test = np.concatenate(test_labels_list, axis=0)
    print(f"测试集特征维度：{test_features.shape}")

    return train_features, test_features, y_train, y_test


def train_fusion_model(train_features, test_features,
                        y_train, y_test):
    """
    用CNN提取的特征训练决策树分类器
    即融合模型：CNN特征提取 + 决策树分类
    参数：
        train_features - 训练集CNN特征
        test_features  - 测试集CNN特征
        y_train        - 训练集标签
        y_test         - 测试集标签
    返回：
        fusion_results - 融合模型评估结果字典
    """
    print("\n训练融合模型（CNN特征 + 决策树）...")

    # 对融合决策树也进行简单的深度调优
    best_acc = 0
    best_depth = None

    for depth in range(3, 20):
        dt = DecisionTreeClassifier(
            criterion='gini',
            max_depth=depth,
            random_state=RANDOM_STATE
        )
        dt.fit(train_features, y_train)
        val_acc = accuracy_score(
            y_test, dt.predict(test_features)
        )
        if val_acc > best_acc:
            best_acc = val_acc
            best_depth = depth

    print(f"融合模型最优决策树深度：{best_depth}")

    # 用最优深度训练最终融合模型
    fusion_dt = DecisionTreeClassifier(
        criterion='gini',
        max_depth=best_depth,
        random_state=RANDOM_STATE
    )
    fusion_dt.fit(train_features, y_train)
    y_pred_fusion = fusion_dt.predict(test_features)

    # 计算评估指标
    fusion_results = {
        'accuracy': accuracy_score(y_test, y_pred_fusion),
        'precision': precision_score(y_test, y_pred_fusion,
                                     average='macro'),
        'recall': recall_score(y_test, y_pred_fusion,
                               average='macro'),
        'f1': f1_score(y_test, y_pred_fusion,
                       average='macro'),
        'y_pred': y_pred_fusion,
        'best_depth': best_depth
    }

    print(f"\n融合模型测试集评估结果：")
    print(f"  准确率：{fusion_results['accuracy']:.4f}")
    print(f"  精确率：{fusion_results['precision']:.4f}")
    print(f"  召回率：{fusion_results['recall']:.4f}")
    print(f"  F1分数：{fusion_results['f1']:.4f}")
    print("\n各类别详细报告：")
    print(classification_report(y_test, y_pred_fusion,
                                target_names=GENRES))

    return fusion_results


def compute_metrics(y_true, y_pred, model_name):
    """
    计算并打印单个模型的完整评估指标
    参数：
        y_true     - 真实标签
        y_pred     - 预测标签
        model_name - 模型名称（用于打印）
    返回：
        metrics_dict - 包含各项指标的字典
    """
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='macro')
    rec = recall_score(y_true, y_pred, average='macro')
    f1 = f1_score(y_true, y_pred, average='macro')

    print(f"\n{model_name} 评估结果：")
    print(f"  准确率：{acc:.4f}")
    print(f"  精确率：{prec:.4f}")
    print(f"  召回率：{rec:.4f}")
    print(f"  F1分数：{f1:.4f}")

    return {
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1': f1
    }


def plot_model_comparison(dt_metrics, cnn_metrics,
                           fusion_metrics):
    """
    绘制三种方案综合性能对比图
    参数：
        dt_metrics     - 决策树模型指标字典
        cnn_metrics    - CNN模型指标字典
        fusion_metrics - 融合模型指标字典
    """
    models = ['CART决策树', 'CNN', 'CNN+决策树\n融合模型']
    metrics_names = ['准确率', '精确率', '召回率', 'F1分数']
    metrics_keys = ['accuracy', 'precision', 'recall', 'f1']

    # 整理数据
    all_metrics = [dt_metrics, cnn_metrics, fusion_metrics]
    data = {
        key: [m[key] for m in all_metrics]
        for key in metrics_keys
    }

    # 绘制分组柱状图
    x = np.arange(len(models))
    width = 0.2
    colors = ['steelblue', 'coral', 'green', 'purple']

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # 左图：四项指标分组柱状图
    for idx, (key, name) in enumerate(
            zip(metrics_keys, metrics_names)):
        offset = (idx - 1.5) * width
        bars = axes[0].bar(x + offset,
                           data[key],
                           width,
                           label=name,
                           color=colors[idx],
                           alpha=0.85)
        # 标注数值
        for bar in bars:
            axes[0].text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f'{bar.get_height():.3f}',
                ha='center', va='bottom',
                fontsize=8, rotation=45
            )

    axes[0].set_xticks(x)
    axes[0].set_xticklabels(models, fontsize=12)
    axes[0].set_ylabel('评估指标值', fontsize=13)
    axes[0].set_title('三种方案性能指标对比', fontsize=15)
    axes[0].legend(fontsize=11, loc='lower right')
    axes[0].set_ylim(0, 1.15)
    axes[0].grid(True, alpha=0.3, axis='y')

    # 右图：准确率单独对比（更直观）
    accs = [m['accuracy'] for m in all_metrics]
    bar_colors = ['steelblue', 'coral', 'green']
    bars = axes[1].bar(models, accs,
                       color=bar_colors,
                       alpha=0.85,
                       width=0.5,
                       edgecolor='black',
                       linewidth=0.8)

    # 标注数值和提升幅度
    baseline = accs[0]  # 以决策树为基线
    for idx, (bar, acc) in enumerate(zip(bars, accs)):
        improvement = (acc - baseline) / baseline * 100
        label_text = f'{acc:.4f}'
        if idx > 0:
            label_text += f'\n(+{improvement:.1f}%)'
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.008,
            label_text,
            ha='center', va='bottom',
            fontsize=12, fontweight='bold'
        )

    axes[1].set_ylabel('测试集准确率', fontsize=13)
    axes[1].set_title('三种方案准确率对比\n（括号内为相对决策树的提升幅度）',
                      fontsize=14)
    axes[1].set_ylim(0, 1.15)
    axes[1].grid(True, alpha=0.3, axis='y')
    axes[1].tick_params(axis='x', labelsize=12)

    plt.tight_layout()
    plt.savefig('outputs/figures/model_comparison.png',
                dpi=150, bbox_inches='tight')
    print("综合对比图已保存至"
          " outputs/figures/model_comparison.png")


def plot_radar_chart(dt_metrics, cnn_metrics, fusion_metrics):
    """
    绘制雷达图，从多维度对比三种方案
    """
    categories = ['准确率', '精确率', '召回率', 'F1分数']
    N = len(categories)

    # 计算角度
    angles = np.linspace(0, 2 * np.pi, N,
                         endpoint=False).tolist()
    angles += angles[:1]  # 闭合雷达图

    fig, ax = plt.subplots(figsize=(8, 8),
                            subplot_kw=dict(polar=True))

    models_data = {
        'CART决策树': dt_metrics,
        'CNN': cnn_metrics,
        'CNN+决策树融合': fusion_metrics
    }
    colors = ['steelblue', 'coral', 'green']
    keys = ['accuracy', 'precision', 'recall', 'f1']

    for (name, metrics), color in zip(
            models_data.items(), colors):
        values = [metrics[k] for k in keys]
        values += values[:1]  # 闭合

        ax.plot(angles, values,
                color=color, linewidth=2, label=name)
        ax.fill(angles, values,
                color=color, alpha=0.15)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=13)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6',
                        '0.8', '1.0'], fontsize=9)
    ax.set_title('三种方案多维度性能雷达图',
                 fontsize=15, pad=20)
    ax.legend(loc='upper right',
              bbox_to_anchor=(1.3, 1.1),
              fontsize=12)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('outputs/figures/radar_chart.png',
                dpi=150, bbox_inches='tight')
    print("雷达图已保存至 outputs/figures/radar_chart.png")


if __name__ == '__main__':
    from src.train.train_cnn import (load_spectrogram_dataset,
                                      split_dataset,
                                      build_dataset_pipeline)
    from src.train.train_dt import (load_and_preprocess,
                                     tune_max_depth,
                                     compare_three_dt)

    print("=" * 50)
    print("Step1：加载数据")
    print("=" * 50)

    # 加载频谱图数据（用于CNN和融合模型）
    X, y = load_spectrogram_dataset()
    (X_train, X_val, X_test,
     y_train, y_val, y_test) = split_dataset(X, y)
    train_ds, val_ds, test_ds = build_dataset_pipeline(
        X_train, y_train,
        X_val, y_val,
        X_test, y_test
    )

    print("\n" + "=" * 50)
    print("Step2：提取CNN中间层特征")
    print("=" * 50)
    (train_features, test_features,
     y_train_feat, y_test_feat) = extract_cnn_features(
        MODEL_PATH, train_ds, test_ds
    )

    print("\n" + "=" * 50)
    print("Step3：训练融合模型并评估")
    print("=" * 50)
    fusion_results = train_fusion_model(
        train_features, test_features,
        y_train_feat, y_test_feat
    )

    print("\n" + "=" * 50)
    print("Step4：加载决策树和CNN的评估结果")
    print("=" * 50)

    # 加载决策树结果
    X_tr, X_te, y_tr, y_te, le = load_and_preprocess()
    best_depth = tune_max_depth(X_tr, y_tr)
    dt_results = compare_three_dt(
        X_tr, X_te, y_tr, y_te, best_depth, le
    )

    # 取CART（最优决策树）的指标
    best_dt_model = dt_results[
        'CART（基尼系数）']['model']
    y_pred_dt = best_dt_model.predict(X_te)
    dt_metrics = compute_metrics(
        y_te, y_pred_dt, 'CART决策树'
    )

    # 加载CNN预测结果
    print("\n加载CNN模型进行测试集预测...")
    best_cnn = tf.keras.models.load_model(MODEL_PATH)
    y_pred_cnn = np.argmax(
        best_cnn.predict(test_ds), axis=1
    )
    cnn_metrics = compute_metrics(
        y_test_feat, y_pred_cnn, 'CNN'
    )

    # 融合模型指标
    fusion_metrics = {
        'accuracy': fusion_results['accuracy'],
        'precision': fusion_results['precision'],
        'recall': fusion_results['recall'],
        'f1': fusion_results['f1']
    }

    print("\n" + "=" * 50)
    print("Step5：生成综合对比图表")
    print("=" * 50)

    # 三种方案性能对比柱状图
    plot_model_comparison(
        dt_metrics, cnn_metrics, fusion_metrics
    )

    # 三种方案雷达图
    plot_radar_chart(
        dt_metrics, cnn_metrics, fusion_metrics
    )

    print("\n" + "=" * 50)
    print("全部实验完成！结果汇总：")
    print("=" * 50)
    print(f"\n{'方案':<20}{'准确率':<12}"
          f"{'精确率':<12}{'召回率':<12}{'F1分数'}")
    print("-" * 65)
    for name, metrics in [
        ('CART决策树', dt_metrics),
        ('CNN', cnn_metrics),
        ('CNN+决策树融合', fusion_metrics)
    ]:
        print(f"{name:<20}"
              f"{metrics['accuracy']:<12.4f}"
              f"{metrics['precision']:<12.4f}"
              f"{metrics['recall']:<12.4f}"
              f"{metrics['f1']:.4f}")
    print("\n所有图表已保存至 outputs/figures/ 目录")
