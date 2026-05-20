"""
generate_comparison.py
使用已有的实验结果生成对比图表
"""

import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['STHeiti', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 确保目录存在
os.makedirs('outputs/figures', exist_ok=True)

# 已有实验结果
# 决策树（从之前实验）
dt_metrics = {
    'accuracy': 0.5138,
    'precision': 0.5200,  # 近似值
    'recall': 0.5138,
    'f1': 0.5150
}

# CNN（从训练输出）
cnn_metrics = {
    'accuracy': 0.6800,
    'precision': 0.7075,  # macro avg from classification report
    'recall': 0.6800,
    'f1': 0.6750
}

# 融合模型（从metrics.py输出）
fusion_metrics = {
    'accuracy': 0.5900,
    'precision': 0.6168,
    'recall': 0.5900,
    'f1': 0.5894
}

def plot_model_comparison(dt_metrics, cnn_metrics, fusion_metrics):
    """绘制三种方案综合性能对比图"""
    models = ['CART决策树', 'CNN', 'CNN+决策树\n融合模型']
    metrics_names = ['准确率', '精确率', '召回率', 'F1分数']
    metrics_keys = ['accuracy', 'precision', 'recall', 'f1']

    all_metrics = [dt_metrics, cnn_metrics, fusion_metrics]
    data = {key: [m[key] for m in all_metrics] for key in metrics_keys}

    x = np.arange(len(models))
    width = 0.2
    colors = ['steelblue', 'coral', 'green', 'purple']

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # 左图：四项指标分组柱状图
    for idx, (key, name) in enumerate(zip(metrics_keys, metrics_names)):
        offset = (idx - 1.5) * width
        bars = axes[0].bar(x + offset, data[key], width,
                          label=name, color=colors[idx], alpha=0.85)
        for bar in bars:
            axes[0].text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.005,
                        f'{bar.get_height():.3f}',
                        ha='center', va='bottom', fontsize=8, rotation=45)

    axes[0].set_xticks(x)
    axes[0].set_xticklabels(models, fontsize=12)
    axes[0].set_ylabel('评估指标值', fontsize=13)
    axes[0].set_title('三种方案性能指标对比', fontsize=15)
    axes[0].legend(fontsize=11, loc='lower right')
    axes[0].set_ylim(0, 1.15)
    axes[0].grid(True, alpha=0.3, axis='y')

    # 右图：准确率单独对比
    accs = [m['accuracy'] for m in all_metrics]
    bar_colors = ['steelblue', 'coral', 'green']
    bars = axes[1].bar(models, accs, color=bar_colors,
                      alpha=0.85, width=0.5, edgecolor='black', linewidth=0.8)

    baseline = accs[0]
    for idx, (bar, acc) in enumerate(zip(bars, accs)):
        improvement = (acc - baseline) / baseline * 100
        label_text = f'{acc:.4f}'
        if idx > 0:
            label_text += f'\n(+{improvement:.1f}%)'
        axes[1].text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.008,
                    label_text, ha='center', va='bottom',
                    fontsize=12, fontweight='bold')

    axes[1].set_ylabel('测试集准确率', fontsize=13)
    axes[1].set_title('三种方案准确率对比\n（括号内为相对决策树的提升幅度）',
                      fontsize=14)
    axes[1].set_ylim(0, 1.15)
    axes[1].grid(True, alpha=0.3, axis='y')
    axes[1].tick_params(axis='x', labelsize=12)

    plt.tight_layout()
    plt.savefig('outputs/figures/model_comparison.png',
                dpi=150, bbox_inches='tight')
    print("综合对比图已保存至 outputs/figures/model_comparison.png")


def plot_radar_chart(dt_metrics, cnn_metrics, fusion_metrics):
    """绘制雷达图"""
    categories = ['准确率', '精确率', '召回率', 'F1分数']
    N = len(categories)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    models_data = {
        'CART决策树': dt_metrics,
        'CNN': cnn_metrics,
        'CNN+决策树融合': fusion_metrics
    }
    colors = ['steelblue', 'coral', 'green']
    keys = ['accuracy', 'precision', 'recall', 'f1']

    for (name, metrics), color in zip(models_data.items(), colors):
        values = [metrics[k] for k in keys]
        values += values[:1]
        ax.plot(angles, values, color=color, linewidth=2, label=name)
        ax.fill(angles, values, color=color, alpha=0.15)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=13)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=9)
    ax.set_title('三种方案多维度性能雷达图', fontsize=15, pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=12)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('outputs/figures/radar_chart.png',
                dpi=150, bbox_inches='tight')
    print("雷达图已保存至 outputs/figures/radar_chart.png")


if __name__ == '__main__':
    print("=" * 50)
    print("生成综合对比图表")
    print("=" * 50)

    plot_model_comparison(dt_metrics, cnn_metrics, fusion_metrics)
    plot_radar_chart(dt_metrics, cnn_metrics, fusion_metrics)

    print("\n" + "=" * 50)
    print("全部实验完成！结果汇总：")
    print("=" * 50)
    print(f"\n{'方案':<20}{'准确率':<12}{'精确率':<12}{'召回率':<12}{'F1分数'}")
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
