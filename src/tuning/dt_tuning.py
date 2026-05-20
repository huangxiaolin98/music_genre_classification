"""
dt_tuning.py
功能：决策树参数调优实验
      对ID3、C4.5、CART分别进行深度调优
      对比不同准则下的最优参数
"""

import numpy as np
import os
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['STHeiti', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import validation_curve

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from src.train.train_dt import load_and_preprocess, RANDOM_STATE, CV_FOLDS

# ============ 全局配置 ============
DEPTH_RANGE = range(1, 21)


def tune_all_dt_algorithms(X_train, y_train):
    """
    对三种决策树算法分别进行深度调优
    返回各算法的最优深度
    """
    algorithms = {
        'ID3（entropy）': {
            'criterion': 'entropy',
            'splitter': 'best',
            'max_features': None
        },
        'C4.5（entropy+sqrt）': {
            'criterion': 'entropy',
            'splitter': 'best',
            'max_features': 'sqrt'
        },
        'CART（gini）': {
            'criterion': 'gini',
            'splitter': 'best',
            'max_features': None
        }
    }

    results = {}
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for idx, (name, params) in enumerate(algorithms.items()):
        print(f"\n正在调优：{name}")

        estimator = DecisionTreeClassifier(
            criterion=params['criterion'],
            splitter=params['splitter'],
            max_features=params['max_features'],
            random_state=RANDOM_STATE
        )

        train_scores, val_scores = validation_curve(
            estimator,
            X_train, y_train,
            param_name='max_depth',
            param_range=DEPTH_RANGE,
            cv=CV_FOLDS,
            scoring='accuracy'
        )

        train_mean = np.mean(train_scores, axis=1)
        val_mean = np.mean(val_scores, axis=1)
        val_std = np.std(val_scores, axis=1)

        best_depth = list(DEPTH_RANGE)[np.argmax(val_mean)]
        best_acc = np.max(val_mean)

        results[name] = {
            'best_depth': best_depth,
            'best_acc': best_acc,
            'train_scores': train_mean,
            'val_scores': val_mean
        }

        print(f"  最优深度：{best_depth}，"
              f"验证准确率：{best_acc:.4f}")

        # 绘制各算法的调优曲线
        axes[idx].plot(DEPTH_RANGE, train_mean, 'b-o',
                       label='训练集', linewidth=2, markersize=4)
        axes[idx].plot(DEPTH_RANGE, val_mean, 'r-o',
                       label='验证集', linewidth=2, markersize=4)
        axes[idx].fill_between(DEPTH_RANGE,
                               val_mean - val_std,
                               val_mean + val_std,
                               alpha=0.1, color='red')
        axes[idx].axvline(x=best_depth, color='green',
                          linestyle='--',
                          label=f'最优={best_depth}')
        axes[idx].set_xlabel('max_depth', fontsize=12)
        axes[idx].set_ylabel('准确率', fontsize=12)
        axes[idx].set_title(f'{name}\n最优深度={best_depth}',
                            fontsize=13)
        axes[idx].legend(fontsize=10)
        axes[idx].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('outputs/figures/dt_all_tuning.png',
                dpi=150, bbox_inches='tight')
    print("\n三种决策树调优曲线已保存至"
          " outputs/figures/dt_all_tuning.png")

    return results


if __name__ == '__main__':
    os.makedirs('outputs/figures', exist_ok=True)

    # 加载数据
    X_train, X_test, y_train, y_test, le = load_and_preprocess()

    # 对三种算法进行深度调优
    results = tune_all_dt_algorithms(X_train, y_train)

    # 输出汇总
    print("\n" + "=" * 50)
    print("决策树参数调优结果汇总：")
    print("=" * 50)
    print(f"{'算法':<25}{'最优深度':<12}{'验证准确率'}")
    print("-" * 50)
    for name, res in results.items():
        print(f"{name:<25}{res['best_depth']:<12}"
              f"{res['best_acc']:.4f}")
