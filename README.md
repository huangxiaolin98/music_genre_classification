# 基于CNN与决策树的音乐流派分类研究

> 模式识别与机器学习课程设计
> 
> Music Genre Classification Based on CNN and Decision Tree

---

## 项目背景与目的

音乐流派分类是音乐信息检索（Music Information Retrieval, MIR）
领域的经典问题，在音乐推荐系统、数字音乐库管理、
个性化播放列表生成等场景中具有重要的实用价值。

传统方法依赖人工提取音频特征（如MFCC、过零率、频谱质心等），
再结合机器学习分类器进行判断；而深度学习方法则通过将音频转换为
梅尔频谱图（Mel Spectrogram），利用卷积神经网络（CNN）
自动学习音频的时频特征表示。

本项目以 GTZAN Music Genre Dataset 为实验数据集，
同时实现以下两条技术路线并进行对比：

- 路线A：手工特征（MFCC等）+ 决策树（ID3 / C4.5 / CART）
- 路线B：梅尔频谱图 + 卷积神经网络（CNN）
- 路线C（创新）：CNN提取深层特征 + 决策树分类器（融合方案）

通过多算法横向对比与系统性参数调优实验，
探究不同方法在音乐流派分类任务上的性能差异。

---

## 工程目录结构

music_genre_classification/
│
├── README.md                    # 本文档
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
│   │   ├── features_30_sec.csv  # GTZAN自带特征CSV
│   │   └── features_3_sec.csv   # 3秒片段特征CSV
│   │
│   └── spectrograms/            # 生成的梅尔频谱图
│       ├── blues/
│       ├── classical/
│       └── .../
│
├── src/                         # 源代码目录
│   ├── preprocess/              # 数据预处理模块
│   │   ├── __init__.py
│   │   ├── audio_to_spectrogram.py   # 音频转频谱图
│   │   └── feature_extraction.py    # 手工特征提取
│   │
│   ├── models/                  # 模型定义模块
│   │   ├── __init__.py
│   │   ├── cnn_model.py         # CNN模型定义
│   │   └── decision_tree.py     # 决策树模型
│   │
│   ├── train/                   # 训练模块
│   │   ├── __init__.py
│   │   ├── train_cnn.py         # CNN训练主程序
│   │   └── train_dt.py          # 决策树训练主程序
│   │
│   ├── tuning/                  # 参数调优模块
│   │   ├── __init__.py
│   │   ├── cnn_tuning.py        # CNN超参数调优
│   │   └── dt_tuning.py         # 决策树参数调优
│   │
│   └── evaluate/                # 评估与可视化模块
│       ├── __init__.py
│       ├── metrics.py           # 评估指标计算
│       └── visualization.py     # 结果可视化
│
├── notebooks/                   # Jupyter Notebook（可选）
│   ├── 01_data_exploration.ipynb
│   ├── 02_decision_tree.ipynb
│   ├── 03_cnn_training.ipynb
│   └── 04_comparison.ipynb
│
├── outputs/                     # 输出结果目录
│   ├── models/                  # 保存的模型文件
│   │   ├── best_cnn_model.h5
│   │   └── best_dt_model.pkl
│   │
│   └── figures/                 # 生成的图表
│       ├── spectrogram_samples.png
│       ├── lr_comparison.png
│       ├── depth_tuning.png
│       ├── confusion_matrix.png
│       └── model_comparison.png
│
├── requirements.txt             # Python依赖包
└── main.py                      # 一键运行入口
