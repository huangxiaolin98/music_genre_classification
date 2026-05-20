

## 实验步骤指南

---

### 前置准备

#### Step 0：下载 GTZAN 数据集

目前 `data/raw/genres_original/` 和 `data/features/` 都是空的，需要先下载数据集。

1. 访问 Kaggle 的 [GTZAN Dataset](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification) 页面下载
2. 解压后，将 `genres_original/` 文件夹放入 `data/raw/genres_original/`（确保包含 10 个子文件夹：blues, classical, country, disco, hiphop, jazz, metal, pop, reggae, rock，每个 100 首 .wav 文件）
3. 将 `features_30_sec.csv` 放入 `data/features/`

#### Step 1：创建虚拟环境并安装依赖

```bash
conda create -n music_cls python=3.9
conda activate music_cls
pip install -r requirements.txt
```

---

### 实验一：决策树路线

#### Step 2：运行决策树实验

```bash
python src/train/train_dt.py
```

这一步动完成：
- 加载 `data/features/features_30_sec.csv`
- 预处理（标准化、分层划分 8:2）
- **深度调优**：遍历 max_depth=1~20，5 折交叉验证，生成调优曲线图
- **三种算法对比**：ID3 / C4.5 / CART，输出准确率、精确率、召回率、F1
- 图表保存至 `outputs/figures/dt_depth_tuning.png` 和 `dt_comparison.png`

**需要记录的数据**：报告 4.1 节中所有 `[待填入]` 的表格数据

---

### 实验二：CNN 路线

#### Step 3：生成梅尔频谱图

```bash
python src/preprocess/audio_to_spectrogram.py
```

- 读取 `data/raw/genres_original/` 下的 1000 个 .wav 文件
- 转换为 128×128 梅尔频谱图 PNG
- 保存至 `data/spectrograms/` 各流派子文件夹
- CPU 下预计 5~10 分钟

#### Step 4：CNN 参数调优

```bash
python src/tuning/cnn_tuning.py
```

依次进行三组调优实验（每组 50 Epoch）：
1. **学习率调优**：0.1 / 0.01 / 0.001 / 0.0001 → 生成 `lr_comparison.png`
2. **网络深度调优**：2 / 3 / 4 卷积块 → 生成 `cnn_structure_tuning.png`
3. **Dropout率调优**：0.2 / 0.3 / 0.5 → 生成 `dropout_tuning.png`

**需要记录的数据**：报告 4.2 节中三组调优的表格数据

#### Step 5：训练最优 CNN 模型

```bash
python src/train/train_cnn.py
```

- 使用调优后的最优超参数组合训练 100 Epoch
- 包含早停、学习率衰减、模型保存回调
- 模型保存至 `outputs/models/best_cnn_model.h5`
- 生成训练曲线 `cnn_training_curve.png` 和混淆矩阵 `confusion_matrix.png`

**需要记录的数据**：报告 4.2.4 节的最优超参数表和测试集性能表、4.3 节各流派详细结果

---

### 实验三：融合模型

#### Step 6：运行融合模型实验

```bash
python src/evaluate/metrics.py
```

- 加载训练好的 CNN 模型
- 提取 `dense_1` 层的 256 维中间特征
- 用 CNN 特征训练 CART 决策树
- 生成三种方案（CART决策树 / CNN / 融合）综合对比图 `model_comparison.png`

**需要记录的数据**：报告 4.4 节的综合对比表和 5.4 节的融合方案对比表

---

### 一键运行（可选）

```bash
python main.py
```

这会依次执行上述所有步骤。

---

### 实验顺序总结

| 顺序 | 步骤 | 命令 | 产出 |
|------|------|------|------|
| 0 | 下载数据集 | 手动下载 | 音频文件 + features CSV |
| 1 | 创建环境 | conda + pip | Python 虚拟环境 |
| 2 | 决策树实验 | `python src/train/train_dt.py` | 调优曲线 + 三算法对比 |
| 3 | 生成频谱图 | `python src/preprocess/audio_to_spectrogram.py` | 1000 张 PNG |
| 4 | CNN 调优 | `python src/tuning/cnn_tuning.py` | 3 组调优图表 |
| 5 | 训练最优 CNN | `python src/train/train_cnn.py` | 模型文件 + 训练曲线 + 混淆矩阵 |
| 6 | 融合模型 | `python src/evaluate/metrics.py` | 三方案对比图 |

> 关键前提：Step 2（决策树）只依赖 `features_30_sec.csv`，与 Step 3（频谱图）无依赖关系，两者可以并行。但 Step 4~6 都依赖 Step 3 的输出，必须顺序执行。

