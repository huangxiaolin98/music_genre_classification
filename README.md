# 基于CNN与决策树的音乐流派分类研究

> 模式识别与机器学习课程设计
> 
> Music Genre Classification Based on CNN and Decision Tree

---

## 项目背景与目的

音乐流派分类是音乐信息检索（Music Information Retrieval, MIR）领域的经典问题，在音乐推荐系统、数字音乐库管理、个性化播放列表生成等场景中具有重要的实用价值。

传统方法依赖人工提取音频特征（如MFCC、过零率、频谱质心等），再结合机器学习分类器进行判断；而深度学习方法则通过将音频转换为梅尔频谱图（Mel Spectrogram），利用卷积神经网络（CNN）自动学习音频的时频特征表示。

本项目以 GTZAN Music Genre Dataset 为实验数据集，同时实现以下三条技术路线并进行对比：

- **路线A**：手工特征（MFCC等）+ 决策树（ID3 / C4.5 / CART）
- **路线B**：梅尔频谱图 + 卷积神经网络（CNN）
- **路线C（创新）**：CNN提取深层特征 + 决策树分类器（融合方案）

通过多算法横向对比与系统性参数调优实验，探究不同方法在音乐流派分类任务上的性能差异。

---

## 数据集

使用 **GTZAN Music Genre Dataset**（George Tzanetakis, 2002），是音乐流派分类领域最广泛使用的基准数据集。

| 属性 | 内容 |
|------|------|
| 音乐流派数量 | 10类 |
| 每类样本数 | 100首 |
| 总样本数 | 1000首 |
| 每首时长 | 30秒 |
| 采样率 | 22050 Hz |
| 格式 | .wav |

10个流派：Blues、Classical、Country、Disco、HipHop、Jazz、Metal、Pop、Reggae、Rock

数据集下载：https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification

---

## 技术方案

### 决策树路线

- 使用数据集提供的 `features_30_sec.csv`（57维手工特征）
- 特征标准化（Z-score）+ 分层划分（8:2）
- 分别实现 ID3（信息增益）、C4.5（信息增益率近似）、CART（基尼系数）
- 5折交叉验证 + max_depth 参数调优（1~20）

### CNN路线

- 音频转128维梅尔频谱图，归一化至 [0,1]，统一为 128x128 灰度图
- 数据集划分 8:1:1（训练/验证/测试），训练集进行数据增强
- 三卷积块 CNN 模型（32→64→128滤波器），含 BatchNorm + MaxPool + Dropout
- 全连接层 256维 + Softmax 10分类输出
- Adam优化器，稀疏类别交叉熵损失
- 系统调优：学习率（0.1/0.01/0.001/0.0001）、网络深度（2/3/4块）、Dropout率（0.2/0.3/0.5）

### 融合方案（创新点）

- 将训练好的CNN作为特征提取器，截取全连接层（dense_1）输出的256维深层特征
- 用CNN提取的特征训练CART决策树分类器
- 兼具CNN的特征表达能力与决策树的可解释性

---

## 工程目录结构

```
music_genre_classification/
│
├── README.md                    # 本文档
├── main.py                      # 一键运行入口
├── requirements.txt             # Python依赖包
│
├── data/                        # 数据目录
│   ├── raw/                     # 原始音频文件
│   │   └── genres_original/     # 解压后的GTZAN数据集
│   │       ├── blues/           # 每类100首30秒音频
│   │       ├── classical/
│   │       ├── country/
│   │       ├── disco/
│   │       ├── hiphop/
│   │       ├── jazz/
│   │       ├── metal/
│   │       ├── pop/
│   │       ├── reggae/
│   │       └── rock/
│   │
│   ├── features/                # 提取的特征文件
│   │   └── features_30_sec.csv  # GTZAN自带特征CSV（57维）
│   │
│   └── spectrograms/            # 生成的梅尔频谱图
│       ├── blues/
│       ├── classical/
│       └── .../
│
├── src/                         # 源代码目录
│   ├── preprocess/              # 数据预处理模块
│   │   ├── __init__.py
│   │   ├── audio_to_spectrogram.py   # 音频转梅尔频谱图
│   │   └── feature_extraction.py     # 手工特征提取（MFCC等）
│   │
│   ├── models/                  # 模型定义模块
│   │   ├── __init__.py
│   │   ├── cnn_model.py         # CNN模型定义（支持可配置卷积块数）
│   │   └── decision_tree.py     # 决策树模型封装（ID3/C4.5/CART）
│   │
│   ├── train/                   # 训练模块
│   │   ├── __init__.py
│   │   ├── train_cnn.py         # CNN训练主程序
│   │   └── train_dt.py          # 决策树训练与对比主程序
│   │
│   ├── tuning/                  # 参数调优模块
│   │   ├── __init__.py
│   │   ├── cnn_tuning.py        # CNN超参数调优（学习率/深度/Dropout）
│   │   └── dt_tuning.py         # 决策树参数调优
│   │
│   └── evaluate/                # 评估与可视化模块
│       ├── __init__.py
│       ├── metrics.py           # 融合模型实验 + 综合性能对比
│       └── visualization.py     # 通用可视化工具函数
│
├── notebooks/                   # Jupyter Notebook（可选）
│
└── outputs/                     # 输出结果目录
    ├── models/                  # 保存的模型文件
    │   └── best_cnn_model.h5    # 最优CNN模型
    │
    └── figures/                 # 生成的图表
        ├── dt_depth_tuning.png       # 决策树深度调优曲线
        ├── dt_comparison.png         # 三种决策树性能对比
        ├── lr_comparison.png         # CNN学习率调优对比
        ├── cnn_structure_tuning.png  # CNN网络深度调优对比
        ├── dropout_tuning.png        # Dropout率调优对比
        ├── cnn_training_curve.png    # 最优CNN训练曲线
        ├── confusion_matrix.png      # CNN混淆矩阵热力图
        ├── model_comparison.png      # 三种方案综合对比
        └── radar_chart.png           # 三种方案雷达图
```

---

## 运行环境

| 环境项目 | 配置信息 |
|----------|----------|
| Python版本 | 3.9 |
| 深度学习框架 | TensorFlow >= 2.10.0 |
| 机器学习库 | scikit-learn >= 1.2.0 |
| 音频处理库 | librosa >= 0.9.2 |
| 数值计算 | numpy >= 1.23.5 |
| 数据处理 | pandas >= 1.5.3 |
| 可视化 | matplotlib >= 3.6.3, seaborn >= 0.12.2 |

---

## 快速开始

### 1. 环境配置

```bash
conda create -n music_cls python=3.9
conda activate music_cls
pip install -r requirements.txt
```

### 2. 数据集准备

从 Kaggle 下载 GTZAN 数据集，解压后放置：

- 将 `genres_original/` 文件夹放入 `data/raw/` 目录
- 将 `features_30_sec.csv` 放入 `data/features/` 目录

### 3. 一键运行全部实验

```bash
python main.py
```

### 4. 分步运行

```bash
# Step1：生成梅尔频谱图
python src/preprocess/audio_to_spectrogram.py

# Step2：决策树实验（训练 + 调优 + 对比）
python src/train/train_dt.py

# Step3：CNN参数调优实验
python src/tuning/cnn_tuning.py

# Step4：训练最优CNN模型
python src/train/train_cnn.py

# Step5：融合模型实验 + 综合性能对比
python src/evaluate/metrics.py
```

---

## 实验内容

1. 决策树 max_depth 参数调优实验（5折交叉验证）
2. ID3 / C4.5 / CART 三种决策树对比实验
3. CNN 学习率调优实验（0.1 / 0.01 / 0.001 / 0.0001）
4. CNN 网络深度调优实验（2 / 3 / 4 个卷积块）
5. Dropout 率调优实验（0.2 / 0.3 / 0.5）
6. CNN + 决策树融合模型实验
7. 三种方案（决策树 / CNN / 融合）综合性能对比

---

## 评价指标

- **准确率（Accuracy）**：分类正确样本占总样本比例
- **精确率（Precision）**：宏平均精确率
- **召回率（Recall）**：宏平均召回率
- **F1分数（F1-Score）**：精确率与召回率的调和平均

所有指标均采用宏平均（Macro Average）方式计算。

---

## 参考文献

1. Tzanetakis G, Cook P. Musical genre classification of audio signals[J]. IEEE TSAP, 2002.
2. LeCun Y, et al. Gradient-based learning applied to document recognition[J]. Proc. IEEE, 1998.
3. Quinlan J R. Induction of decision trees[J]. Machine Learning, 1986.
4. Quinlan J R. C4.5: Programs for Machine Learning[M]. Morgan Kaufmann, 1993.
5. Breiman L, et al. Classification and Regression Trees[M]. CRC Press, 1984.
6. McFee B, et al. librosa: Audio and music signal analysis in python[C]. SciPy, 2015.
7. Kingma D P, Ba J. Adam: A method for stochastic optimization[C]. ICLR, 2015.
8. Srivastava N, et al. Dropout: A simple way to prevent neural networks from overfitting[J]. JMLR, 2014.
9. Ioffe S, Szegedy C. Batch normalization[C]. ICML, 2015.
10. 周志华. 机器学习[M]. 清华大学出版社, 2016.
