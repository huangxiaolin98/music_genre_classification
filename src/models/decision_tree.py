"""
decision_tree.py
功能：决策树模型封装
      支持ID3、C4.5、CART三种决策树配置
"""

from sklearn.tree import DecisionTreeClassifier


def build_decision_tree(algorithm='cart', max_depth=None,
                        random_state=42):
    """
    构建决策树模型
    参数：
        algorithm    - 算法类型：'id3', 'c45', 'cart'
        max_depth    - 树的最大深度
        random_state - 随机种子
    返回：
        sklearn DecisionTreeClassifier 实例
    """
    if algorithm == 'id3':
        # ID3：使用信息增益（entropy）
        model = DecisionTreeClassifier(
            criterion='entropy',
            splitter='best',
            max_depth=max_depth,
            random_state=random_state
        )
    elif algorithm == 'c45':
        # C4.5：使用信息增益率（近似实现）
        model = DecisionTreeClassifier(
            criterion='entropy',
            max_features='sqrt',
            max_depth=max_depth,
            random_state=random_state
        )
    elif algorithm == 'cart':
        # CART：使用基尼系数
        model = DecisionTreeClassifier(
            criterion='gini',
            splitter='best',
            max_depth=max_depth,
            random_state=random_state
        )
    else:
        raise ValueError(f"不支持的算法类型：{algorithm}，"
                         f"可选：'id3', 'c45', 'cart'")

    return model
