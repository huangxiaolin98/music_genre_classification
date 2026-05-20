"""
train_dt.py
功能：训练ID3、C4.5、CART三种决策树模型
      并进行max_depth参数调优
      输出性能对比结果与调优曲线图
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['STHeiti', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
import seaborn as sns
import time
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import (train_test_split,
                                     cross_val_score,
                                     validation_curve)
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (accuracy_score,
                             classification_report,
                             confusion_matrix)
import warnings
warnings.filterwarnings('ignore')

# ============ 全局配置 ============
FEATURE_CSV = "data/features/features_30_sec.csv"
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5    # 交叉验证折数


def load_and_preprocess():
    """
    加载特征CSV文件并进行预处理
    返回：X_train, X_test, y_train, y_test
    """
    df = pd.read_csv(FEATURE_CSV)

    # 删除无关列
    df = df.drop(columns=['filename', 'length'], errors='ignore')

    # 标签编码
    le = LabelEncoder()
    df['label'] = le.fit_transform(df['label'])

    X = df.drop(columns=['label']).values
    y = df['label'].values

    # 特征标准化
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # 分层划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y    # 保证各类别比例均衡
    )

    print(f"训练集大小：{X_train.shape}")
    print(f"测试集大小：{X_test.shape}")
    print(f"特征维度：{X_train.shape[1]}")

    return X_train, X_test, y_train, y_test, le


def tune_max_depth(X_train, y_train):
    """
    对CART决策树进行max_depth参数调优
    使用验证曲线方法，绘制深度vs准确率曲线
    """
    depth_range = range(1, 21)

    # 使用sklearn的validation_curve
    train_scores, val_scores = validation_curve(
        DecisionTreeClassifier(criterion='gini',
                               random_state=RANDOM_STATE),
        X_train, y_train,
        param_name='max_depth',
        param_range=depth_range,
        cv=CV_FOLDS,
        scoring='accuracy'
    )

    # 计算均值和标准差
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)

    # 找出最优深度
    best_depth = list(depth_range)[np.argmax(val_mean)]
    best_val_acc = np.max(val_mean)
    print(f"最优树深度：{best_depth}，"
          f"对应验证集准确率：{best_val_acc:.4f}")

    # 绘制调优曲线
    plt.figure(figsize=(10, 6))
    plt.plot(depth_range, train_mean, 'b-o',
             label='训练集准确率', linewidth=2)
    plt.fill_between(depth_range,
                     train_mean - train_std,
                     train_mean + train_std,
                     alpha=0.1, color='blue')
    plt.plot(depth_range, val_mean, 'r-o',
             label='验证集准确率', linewidth=2)
    plt.fill_between(depth_range,
                     val_mean - val_std,
                     val_mean + val_std,
                     alpha=0.1, color='red')

    # 标记最优点
    plt.axvline(x=best_depth, color='green',
                linestyle='--', label=f'最优深度={best_depth}')
    plt.xlabel('树的最大深度 (max_depth)', fontsize=13)
    plt.ylabel('准确率', fontsize=13)
    plt.title('决策树深度调优曲线（5折交叉验证）', fontsize=15)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('outputs/figures/dt_depth_tuning.png',
                dpi=150, bbox_inches='tight')
    print("深度调优曲线已保存至 outputs/figures/dt_depth_tuning.png")

    return best_depth


def compare_three_dt(X_train, X_test, y_train, y_test,
                     best_depth, label_encoder):
    """
    对比 ID3 / C4.5 / CART 三种决策树的性能
    """
    models = {
        'ID3（信息增益）': DecisionTreeClassifier(
            criterion='entropy',
            splitter='best',
            max_depth=best_depth,
            random_state=RANDOM_STATE
        ),
        'C4.5（信息增益率）': DecisionTreeClassifier(
            criterion='entropy',
            max_features='sqrt',
            max_depth=best_depth,
            random_state=RANDOM_STATE
        ),
        'CART（基尼系数）': DecisionTreeClassifier(
            criterion='gini',
            splitter='best',
            max_depth=best_depth,
            random_state=RANDOM_STATE
        )
    }

    results = {}
    for name, model in models.items():
        # 训练模型（计时）
        start_time = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - start_time

        # 测试集预测
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        # 5折交叉验证
        cv_scores = cross_val_score(
            model, X_train, y_train,
            cv=CV_FOLDS, scoring='accuracy'
        )

        results[name] = {
            'model': model,
            'test_acc': acc,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'train_time': train_time
        }

        print(f"\n{name}")
        print(f"  测试集准确率：{acc:.4f}")
        print(f"  交叉验证准确率：{cv_scores.mean():.4f}"
              f" ± {cv_scores.std():.4f}")
        print(f"  训练时间：{train_time:.4f}s")
        print(classification_report(
            y_test, y_pred,
            target_names=label_encoder.classes_
        ))

    # 绘制三种算法对比柱状图
    names = list(results.keys())
    test_accs = [results[n]['test_acc'] for n in names]
    cv_means = [results[n]['cv_mean'] for n in names]

    x = np.arange(len(names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width/2, test_accs, width,
                   label='测试集准确率', color='steelblue')
    bars2 = ax.bar(x + width/2, cv_means, width,
                   label='交叉验证准确率', color='coral')

    # 在柱子上方标注数值
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.005,
                f'{bar.get_height():.3f}',
                ha='center', va='bottom', fontsize=11)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.005,
                f'{bar.get_height():.3f}',
                ha='center', va='bottom', fontsize=11)

    ax.set_xlabel('决策树算法', fontsize=13)
    ax.set_ylabel('准确率', fontsize=13)
    ax.set_title('三种决策树算法性能对比', fontsize=15)
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=11)
    ax.legend(fontsize=12)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig('outputs/figures/dt_comparison.png',
                dpi=150, bbox_inches='tight')
    print("对比图已保存至 outputs/figures/dt_comparison.png")

    return results


if __name__ == '__main__':
    # 加载数据
    X_train, X_test, y_train, y_test, le = load_and_preprocess()

    # 决策树深度调优
    best_depth = tune_max_depth(X_train, y_train)

    # 三种决策树对比
    results = compare_three_dt(X_train, X_test,
                                y_train, y_test,
                                best_depth, le)
