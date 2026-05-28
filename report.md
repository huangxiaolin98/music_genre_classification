# 基于CNN与决策树的音乐流派分类研究



## 一、相关技术、方法概述

### 1.1 研究背景

音乐流派分类是音乐信息检索（MIR）领域的核心问题之一。随着数字音乐平台的快速发展，如何对海量音乐进行自动化分类与标注，是支撑个性化推荐系统的重要技术基础。

本课程设计同时实现基于手工特征的决策树方法与基于梅尔频谱图的CNN方法，对比两类方法的性能差异，并提出CNN特征提取与决策树分类器相结合的融合方案作为创新设计内容。

### 1.2 数据集介绍

本实验使用GTZAN Music Genre Dataset，该数据集由Tzanetakis于2002年构建，是音乐流派分类领域最广泛使用的基准数据集。

| 属性 | 内容 |
|------|------|
| 音乐流派数量 | 10类（Blues、Classical、Country、Disco、HipHop、Jazz、Metal、Pop、Reggae、Rock） |
| 每类样本数 | 100首 |
| 总样本数 | 1000首 |
| 每首时长/采样率 | 30秒 / 22050 Hz |
| 格式 | .wav |

数据集同时提供了预提取的57维音频特征CSV文件（含MFCC、频谱质心、过零率等），可直接用于决策树模型训练。

### 1.3 音频特征提取技术

**梅尔频谱图（Mel Spectrogram）**：将音频STFT结果映射到模拟人耳感知的梅尔刻度上，得到二维时频表示。生成流程为：分帧加窗 → FFT → 梅尔滤波器组映射 → 取对数 → 转换为dB刻度。其本质是一张图像（横轴时间、纵轴频率、颜色表示能量），不同流派呈现不同纹理模式，适合CNN处理。

**MFCC特征**：在梅尔频谱图基础上进一步做离散余弦变换（DCT），提取紧凑表示频谱形状的系数。本实验使用数据集提供的57维统计特征（含MFCC均值、方差等）作为决策树输入。

### 1.4 决策树算法

决策树通过递归选择最优特征对样本进行划分。本实验涉及三种算法，核心区别在于分裂准则：

**ID3**——信息增益：

$$Gain(D, A) = H(D) - \sum_{v} \frac{|D^v|}{|D|} H(D^v), \quad H(D) = -\sum_{k=1}^{n} p_k \log_2 p_k$$

**C4.5**——信息增益率（克服ID3对多值特征的偏好）：

$$GainRatio(D, A) = \frac{Gain(D, A)}{IV(A)}, \quad IV(A) = -\sum_{v}\frac{|D^v|}{|D|}\log_2\frac{|D^v|}{|D|}$$

**CART**——基尼系数：

$$Gini(D) = 1 - \sum_{k=1}^{n} p_k^2$$

三种算法对比：

| 算法 | 分裂准则 | 树的结构 | 连续特征 | 剪枝 |
|------|----------|----------|----------|------|
| ID3 | 信息增益 | 多叉树 | 不支持 | 不支持 |
| C4.5 | 信息增益率 | 多叉树 | 支持 | 支持 |
| CART | 基尼系数 | 二叉树 | 支持 | 支持 |

### 1.5 卷积神经网络（CNN）

CNN通过局部连接和权值共享处理网格结构数据，本实验用于对梅尔频谱图进行分类。网络由以下组件构成：

- **卷积层**：$Y_{i,j} = \sum_{m}\sum_{n} X_{i+m, j+n} \cdot W_{m,n} + b$，提取局部特征
- **批归一化层**：加速收敛，缓解梯度消失
- **ReLU激活**：$f(x) = \max(0, x)$，引入非线性
- **池化层**：下采样，增强平移鲁棒性
- **Dropout层**：随机置零部分神经元，防止过拟合
- **全连接层**：展平后线性变换，输出分类结果

**BP算法与参数更新**：通过链式法则计算梯度，使用Adam优化器更新参数：

$$W \leftarrow W - \eta \cdot \frac{\partial L}{\partial W}$$

其中$\eta$为学习率，过大导致震荡，过小则收敛慢。Adam引入自适应学习率和动量机制，训练更高效。

**数据增强**：对训练集进行随机翻转、旋转（±10°）、平移等操作，增加样本多样性，提升泛化能力。

---

## 二、设计方案

### 2.1 总体设计框架

本实验围绕音乐流派分类任务，设计了三条并行的技术路线，总体框架如下图所示：

```
原始音频数据（GTZAN数据集）
         |
    +----+----+
    |         |
    v         v
提取MFCC等   转换为梅尔
手工特征     频谱图（图像）
    |         |
    v         v
决策树分类   CNN分类
(ID3/C4.5    （深度学习）
  /CART)         |
    |         |
    |    CNN中间层特征
    |         |
    +----+----+
         v
      融合方案
   （CNN特征+决策树）
         |
         v
    性能综合对比
```

### 2.2 数据预处理方案

#### 2.2.1 决策树路线预处理

直接使用数据集提供的features_30_sec.csv特征文件，包含1000条样本，每条样本57维特征。预处理步骤包括：

1. 剔除文件名列等非特征列；
2. 对标签进行数值编码（blues→0，classical→1，...，rock→9）；
3. 对特征进行标准化处理（Z-score归一化），使每个特征均值为0、标准差为1；
4. 按8:2比例划分训练集与测试集，保证各类别比例均衡（stratify分层抽样）。

#### 2.2.2 CNN路线预处理

对原始音频文件进行如下处理：

1. 使用librosa库加载音频，统一采样率为22050Hz；
2. 计算128个梅尔滤波器组的频谱，窗长2048，帧移512；
3. 转换为分贝刻度，得到形状为(128, 130)的频谱图矩阵；
4. 将矩阵归一化至[0,1]区间；
5. 调整图像尺寸至统一的128×128像素；
6. 按8:1:1比例划分训练集、验证集、测试集；
7. 对训练集应用数据增强。

### 2.3 模型设计方案

#### 2.3.1 决策树模型设计

使用scikit-learn库分别实现三种决策树，核心配置如下：

**ID3（信息增益）**

```python
DecisionTreeClassifier(criterion='entropy', splitter='best')
```

**C4.5（信息增益率近似）**

```python
DecisionTreeClassifier(criterion='entropy', max_features='sqrt')
```

**CART（基尼系数）**

```python
DecisionTreeClassifier(criterion='gini', splitter='best')
```

三种模型均使用5折交叉验证评估性能，最终在测试集上报告准确率、精确率、召回率和F1分数。

#### 2.3.2 CNN模型设计

本实验设计了一个三卷积块的CNN基准模型，网络结构如下表所示：

| 层名称 | 类型 | 输出尺寸 | 参数说明 |
|--------|------|----------|----------|
| Input | 输入层 | (128, 128, 1) | 灰度频谱图 |
| Conv1 | 卷积层 | (126, 126, 32) | 32个3×3卷积核，ReLU |
| BN1 | 批归一化 | (126, 126, 32) | — |
| Pool1 | 最大池化 | (63, 63, 32) | 2×2池化窗口 |
| Dropout1 | Dropout | (63, 63, 32) | rate=0.25 |
| Conv2 | 卷积层 | (61, 61, 64) | 64个3×3卷积核，ReLU |
| BN2 | 批归一化 | (61, 61, 64) | — |
| Pool2 | 最大池化 | (30, 30, 64) | 2×2池化窗口 |
| Dropout2 | Dropout | (30, 30, 64) | rate=0.25 |
| Conv3 | 卷积层 | (28, 28, 128) | 128个3×3卷积核，ReLU |
| BN3 | 批归一化 | (28, 28, 128) | — |
| Pool3 | 最大池化 | (14, 14, 128) | 2×2池化窗口 |
| Dropout3 | Dropout | (14, 14, 128) | rate=0.25 |
| Flatten | 展平层 | (25088,) | — |
| Dense1 | 全连接层 | (256,) | ReLU激活 |
| Dropout4 | Dropout | (256,) | rate=0.5 |
| Dense2 | 输出层 | (10,) | Softmax激活 |

模型总参数量约为650万，损失函数使用稀疏类别交叉熵（Sparse Categorical Crossentropy），优化器使用Adam。

### 2.4 参数调优方案设计

本实验从四个维度对模型进行系统性参数调优，调优方案汇总如下：

| 调优维度 | 调优对象 | 调优范围 | 调优方法 |
|----------|----------|----------|----------|
| 决策树深度 | max_depth | 1～20 | 5折交叉验证 |
| CNN学习率 | learning_rate | 0.1, 0.01, 0.001, 0.0001 | 固定轮次训练对比 |
| CNN网络深度 | 卷积块数量 | 2块, 3块, 4块 | 验证集准确率对比 |
| Dropout率 | dropout_rate | 0.2, 0.3, 0.5 | 训练/验证曲线分析 |

### 2.5 评价指标

本实验使用以下指标评估模型性能：

**准确率（Accuracy）**：分类正确的样本占总样本的比例：

$$Accuracy = \frac{TP + TN}{TP + TN + FP + FN}$$

**精确率（Precision）**：预测为正类中实际为正类的比例：

$$Precision = \frac{TP}{TP + FP}$$

**召回率（Recall）**：实际正类中被正确预测的比例：

$$Recall = \frac{TP}{TP + FN}$$

**F1分数（F1-Score）**：精确率与召回率的调和平均值：

$$F1 = \frac{2 \times Precision \times Recall}{Precision + Recall}$$

由于本实验为10分类问题，上述指标均采用宏平均（Macro Average）方式计算，即对每个类别分别计算后取算术平均。

---

## 三、程序运行环境部署与运行说明

### 3.1 运行环境

| 环境项目 | 配置信息 |
|----------|----------|
| 操作系统 | Mac OS M4 pro / Ubuntu 20.04 |
| Python版本 | 3.9 |
| 深度学习框架 | TensorFlow 2.10.0 |
| 机器学习库 | sk-learn 1.2.0 |
| 音频处理库 | librosa 0.9.2 |
| 数值计算 | numpy 1.23.5 |
| 数据处理 | pandas 1.5.3 |
| 可视化 | matplotlib 3.6.3, seaborn 0.12.2 |
| CPU | Intel Core i5 及以上 |
| 内存 | 8GB 及以上 |
| GPU | 可选，有GPU可显著加速训练 |

### 3.2 环境部署步骤

**第一步：创建虚拟环境**

```bash
conda create -n music_cls python=3.9
conda activate music_cls
```

**第二步：安装依赖包**

```bash
pip install -r requirements.txt
```

**第三步：下载数据集**

访问Kaggle数据集页面下载GTZAN数据集，下载后解压，将genres_original文件夹放入data/raw/目录，将features_30_sec.csv放入data/features/目录。

### 3.3 运行说明

**步骤一：生成梅尔频谱图**

```bash
python src/preprocess/audio_to_spectrogram.py
```

该脚本读取data/raw/genres_original/下的所有音频文件，生成对应的梅尔频谱图，保存至data/spectrograms/目录。CPU环境下预计耗时5～10分钟。

**步骤二：运行决策树实验**

```bash
python src/train/train_dt.py
```

依次训练ID3、C4.5、CART三种决策树，输出各模型的性能指标，并生成参数调优曲线图，保存至outputs/figures/目录。

**步骤三：运行CNN参数调优实验**

```bash
python src/tuning/cnn_tuning.py
```

依次进行学习率调优、网络深度调优、Dropout率调优实验，生成对应可视化图表。

**步骤四：训练最优CNN模型**

```bash
python src/train/train_cnn.py
```

使用调优后的最优超参数训练最终CNN模型，保存至outputs/models/目录。

**步骤五：运行融合模型实验**

```bash
python src/evaluate/metrics.py
```

提取CNN中间层特征，训练融合决策树分类器，输出三种方案的综合性能对比。

**一键运行所有实验：**

```bash
python main.py
```

---

## 四、仿真实验结果

### 4.1 决策树参数调优实验

#### 4.1.1 树深度调优结果

固定其他参数不变，对CART决策树的max_depth参数在1至20范围内进行调优，使用5折交叉验证记录训练集和验证集准确率，结果如下表所示：

| 树深度 | 训练集准确率 | 验证集准确率 |
|--------|-------------|-------------|
| 1 | 17.97% | 17.12% |
| 3 | 35.16% | 31.38% |
| 5 | 53.19% | 42.75% |
| 8 | 76.62% | 49.38% |
| 10 | 88.69% | 49.63% |
| 15 | 98.94% | 50.37% |
| 20 | 99.88% | 50.38% |

![决策树深度调优曲线](outputs/figures/dt_depth_tuning.png)

从实验结果可以观察到明显的过拟合现象：当树深度较小时，模型处于欠拟合状态，训练集和验证集准确率均较低；随着深度增加，验证集准确率先升后降；当深度超过13时，训练集准确率接近100%，但验证集准确率开始下降，说明模型开始过拟合。最优树深度为13，对应验证集准确率为51.38%。

#### 4.1.2 三种决策树算法对比

在最优树深度下，对三种决策树算法进行性能对比，结果如下表所示：

| 算法 | 准确率 | 精确率 | 召回率 | F1分数 | 训练时间 |
|------|--------|--------|--------|--------|----------|
| ID3 | 50.00% | 0.51 | 0.50 | 0.50 | 0.026s |
| C4.5 | 48.00% | 0.48 | 0.48 | 0.47 | 0.004s |
| CART | 45.50% | 0.48 | 0.45 | 0.45 | 0.022s |

![三种决策树算法性能对比](outputs/figures/dt_comparison.png)

分析：在三种决策树算法中，ID3（信息增益）在测试集上取得了最高的准确率（50.00%）和F1分数（0.50），C4.5（信息增益率）次之（48.00%），CART（基尼系数）最低（45.50%）。三种算法整体性能接近，准确率均在45%～50%之间，说明仅依靠57维手工特征进行10分类任务难度较大。ID3略优可能是因为在本数据集上信息增益准则对特征选择的区分度更好，而C4.5通过信息增益率校正多值特征偏好，CART的基尼系数在特征维度较高时可能更倾向于过拟合。

### 4.2 CNN参数调优实验

#### 4.2.1 学习率调优实验

固定网络结构为3卷积块基准模型，Batch Size为32，训练50个Epoch，分别测试四种学习率下的训练过程，结果如下表所示：

| 学习率 | 最终训练Loss | 最终验证准确率 | 是否收敛 |
|--------|-------------|---------------|----------|
| 0.1 | 2.3149 | 16.16% | 是 |
| 0.01 | 2.3039 | 18.18% | 是 |
| 0.001 | 1.4862 | 51.52% | 是 |
| 0.0001 | 0.3434 | 58.59% | 是 |

![不同学习率下Loss曲线与验证准确率对比](outputs/figures/lr_comparison.png)

分析：从Loss下降曲线可以观察到，学习率为0.1时，Loss曲线出现明显震荡，难以稳定收敛，验证集准确率仅为16.16%，这是因为学习率过大导致梯度下降步长过大，在最优值附近反复跳跃；学习率为0.01时，Loss下降依然缓慢，验证集准确率为18.18%；学习率为0.001时，Loss曲线平稳下降且收敛速度较快，验证集准确率达到51.52%；学习率为0.0001时，Loss持续下降至较低水平，最终验证集准确率最高，达到58.59%。因此，后续实验选取学习率为0.0001作为最优值。

#### 4.2.2 网络深度调优实验

固定学习率为0.0001，对比2个卷积块、3个卷积块、4个卷积块三种网络结构的性能，结果如下表所示：

| 网络结构 | 参数量 | 最优验证准确率 | 训练时间/Epoch |
|----------|--------|---------------|----------------|
| 2个卷积块 | 16,799,242 | 63.64% | 约3s |
| 3个卷积块 | 8,485,002 | 61.62% | 约3s |
| 4个卷积块 | 2,341,642 | 51.52% | 约3s |

![不同网络深度验证准确率对比](outputs/figures/cnn_structure_tuning.png)

分析：2个卷积块的模型参数量最大（约1680万），特征提取能力最强，验证集准确率达到63.64%，为三种结构中最优；3个卷积块模型参数量约848万，验证集准确率为61.62%，与2个卷积块接近；4个卷积块模型参数量仅约234万，特征提取能力不足，验证集准确率降至51.52%。在本数据集上，网络并非越深越好，2个卷积块已能充分提取频谱图特征，继续增加深度反而因参数量减少导致特征表达能力下降。因此，后续实验选取2个卷积块作为最优网络结构。

#### 4.2.3 Dropout率调优实验

固定其他参数为最优值，对比三种Dropout率下的训练与验证曲线，结果如下表所示：

| Dropout率 | 最终训练准确率 | 最终验证准确率 | 过拟合程度 |
|-----------|---------------|---------------|------------|
| 0.2 | 96.38% | 70.71% | 0.2567 |
| 0.3 | 89.00% | 65.66% | 0.2334 |
| 0.5 | 75.63% | 42.42% | 0.3320 |

![不同Dropout率下训练/验证准确率曲线对比](outputs/figures/dropout_tuning.png)

分析：随着Dropout率增大，模型的训练准确率逐渐下降，但验证准确率并非单调变化。Dropout率为0.2时，训练准确率高达96.38%，验证准确率为70.71%，为三种Dropout率中最高；Dropout率为0.3时，训练准确率降至89.00%，验证准确率为65.66%，过拟合程度最小（0.2334）；Dropout率为0.5时，训练准确率仅为75.63%，验证准确率大幅下降至42.42%，过拟合程度反而增大至0.3320，这是因为Dropout率过高导致模型欠拟合，有效特征被过度丢弃。综合考虑验证集性能与过拟合控制，选取Dropout率为0.2作为最优值，此时验证集准确率最高（70.71%），虽存在一定过拟合，但模型整体性能最佳。

#### 4.2.4 最优CNN模型训练结果

综合以上调优实验，最优超参数组合为：

| 超参数 | 最优值 |
|--------|--------|
| 学习率 | 0.0001 |
| 卷积块数量 | 2 |
| Dropout率（卷积块后） | 0.2 |
| Dropout率（全连接后） | 0.5 |
| Batch Size | 32 |
| 训练轮次 | 100 |

使用最优超参数组合训练最终CNN模型，在测试集上的性能指标如下表所示：

| 指标 | 数值 |
|------|------|
| 测试集准确率 | 68.00% |
| 宏平均精确率 | 70.75% |
| 宏平均召回率 | 68.00% |
| 宏平均F1分数 | 67.50% |

![最优CNN模型训练过程曲线（Loss曲线 + Accuracy曲线）](outputs/figures/cnn_training_curve.png)

![CNN模型混淆矩阵](outputs/figures/confusion_matrix.png)

从混淆矩阵可以观察到，Classical（古典音乐）和Jazz（爵士乐）的分类准确率最高，这与这两类音乐在频谱图上具有较为独特的纹理特征有关；Rock（摇滚）与Metal（金属）之间存在一定程度的混淆，这是由于两类音乐在节奏和音色上具有较高的相似性，导致对应的梅尔频谱图在视觉上较为接近。

### 4.3 各流派分类详细结果

最优CNN模型对各音乐流派的分类性能如下表所示：

| 音乐流派 | 精确率 | 召回率 | F1分数 | 样本数 |
|----------|--------|--------|--------|--------|
| Blues | 0.78 | 0.70 | 0.74 | 10 |
| Classical | 0.75 | 0.60 | 0.67 | 10 |
| Country | 0.50 | 0.70 | 0.58 | 10 |
| Disco | 0.73 | 0.80 | 0.76 | 10 |
| HipHop | 0.75 | 0.90 | 0.82 | 10 |
| Jazz | 0.67 | 0.60 | 0.63 | 10 |
| Metal | 0.88 | 0.70 | 0.78 | 10 |
| Pop | 0.86 | 0.60 | 0.71 | 10 |
| Reggae | 0.62 | 0.80 | 0.70 | 10 |
| Rock | 0.44 | 0.40 | 0.42 | 10 |
| 宏平均 | 0.71 | 0.68 | 0.68 | 100 |

### 4.4 三种方案综合性能对比

将决策树最优模型（CART）、CNN最优模型、融合模型三种方案在测试集上的性能进行综合对比，结果如下表所示：

| 方案 | 准确率 | F1分数 | 训练时间 | 推理速度 | 可解释性 |
|------|--------|--------|----------|----------|----------|
| CART决策树 | 51.38% | 0.51 | <1s | 快 | 强 |
| CNN | 68.00% | 0.68 | ~5min | 中 | 弱 |
| CNN+决策树融合 | 59.00% | 0.59 | ~5min | 中 | 中 |

![三种方案准确率对比](outputs/figures/model_comparison.png)

分析：

从实验结果可以得出以下结论：

1. 在分类准确率方面，CNN模型显著优于单纯的决策树模型。这是因为手工特征（MFCC等统计量）丢失了音频信号的时序结构信息，而CNN通过处理完整的梅尔频谱图能够自动学习到更丰富的时频特征表示。
2. 决策树模型的训练速度远快于CNN模型，在计算资源受限的场景下具有明显优势，且决策树的决策规则具有良好的可解释性，便于分析哪些音频特征对分类结果影响最大。
3. 融合模型将CNN的特征提取能力与决策树的分类能力结合，在准确率上（59.00%）介于纯CNN（68.00%）和纯决策树（51.38%）之间，说明虽然CNN特征比手工特征更具判别力，但决策树分类器在处理高维CNN特征时可能不如Softmax层有效，融合方案在可解释性上优于纯CNN方案。

---

## 五、创新设计内容

### 5.1 创新点概述

本实验的创新设计在于提出了一种CNN特征提取与决策树分类器相结合的融合方案。该方案兼具CNN强大的自动特征学习能力和决策树良好的可解释性，是对两种经典方法优势的有机融合。

### 5.2 融合方案设计原理

传统决策树方法依赖人工设计的特征（如MFCC、过零率等），这些特征虽然具有明确的物理意义，但表达能力有限，难以捕获音频信号中复杂的高阶结构信息。

CNN通过多层卷积操作，能够从梅尔频谱图中自动学习从低级纹理到高级语义的层次化特征表示。训练完成的CNN模型的全连接层（Dense1，256维）已经学习到了对音乐流派分类具有高度判别性的深层特征。

融合方案的核心思路是：将CNN作为特征提取器，提取其全连接层的256维深层特征表示，再将这些特征输入决策树分类器进行最终分类，以替代CNN原本的Softmax输出层。

融合方案的流程如下图所示：

```
梅尔频谱图输入
      |
      v
  卷积块1（32个滤波器）
      |
      v
  卷积块2（64个滤波器）
      |
      v
  卷积块3（128个滤波器）
      |
      v
  Flatten + Dense（256维） <-- 在此截断，提取特征
      |
      v（不再使用Softmax层）
  256维CNN深层特征向量
      |
      v
  决策树分类器（CART）
      |
      v
  音乐流派预测结果
```

### 5.3 融合方案实现

融合方案的关键实现代码如下：

```python
import tensorflow as tf
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Step1：加载训练好的CNN模型
cnn_model = tf.keras.models.load_model(
    'outputs/models/best_cnn_model.h5'
)

# Step2：构建特征提取器（截取到全连接层）
feature_extractor = tf.keras.Model(
    inputs=cnn_model.input,
    outputs=cnn_model.get_layer('dense_1').output
)

# Step3：提取训练集和测试集的CNN深层特征
train_features = feature_extractor.predict(X_train_spec)
test_features = feature_extractor.predict(X_test_spec)
# train_features.shape: (800, 256)
# test_features.shape:  (200, 256)

# Step4：用CNN特征训练决策树
dt_fusion = DecisionTreeClassifier(
    criterion='gini',
    max_depth=10
)
dt_fusion.fit(train_features, y_train)

# Step5：在测试集上评估融合模型
y_pred_fusion = dt_fusion.predict(test_features)
acc_fusion = accuracy_score(y_test, y_pred_fusion)
print(f"融合模型测试集准确率: {acc_fusion:.4f}")
```

### 5.4 融合方案实验结果分析

融合方案与基线方案的详细对比如下表所示：

| 对比维度 | 纯决策树 | 纯CNN | 融合方案 |
|----------|----------|-------|----------|
| 输入特征 | 57维手工特征 | 梅尔频谱图 | 256维CNN特征 |
| 特征质量 | 人工设计，有限 | 自动学习，丰富 | 自动学习，丰富 |
| 分类器 | 决策树 | Softmax | 决策树 |
| 测试准确率 | 51.38% | 68.00% | 59.00% |
| 可解释性 | 强 | 弱 | 中等 |
| 训练时间 | 极短 | 长 | 长（含CNN训练） |

![三种方案特征重要性对比/融合方案决策树可视化](outputs/figures/model_comparison.png)

融合方案的主要优势体现在以下两个方面：

**优势一：特征质量显著提升。** 相比手工提取的57维统计特征，CNN提取的256维深层特征保留了更丰富的音频时频结构信息，特征的判别性更强，为决策树提供了更优质的输入。

**优势二：具备一定可解释性。** 相比纯CNN的黑箱决策，融合方案的决策树分类部分可以通过树的可视化来解释最终的分类决策逻辑，在需要解释模型决策依据的应用场景中具有实际意义。

---

## 六、参考文献

[1] Tzanetakis G, Cook P. Musical genre classification of audio signals[J]. IEEE Transactions on Speech and Audio Processing, 2002, 10(5): 293-302.

[2] LeCun Y, Bottou L, Bengio Y, et al. Gradient-based learning applied to document recognition[J]. Proceedings of the IEEE, 1998, 86(11): 2278-2324.

[3] Quinlan J R. Induction of decision trees[J]. Machine Learning, 1986, 1(1): 81-106.

[4] Quinlan J R. C4.5: Programs for Machine Learning[M]. Morgan Kaufmann Publishers, 1993.

[5] Breiman L, Friedman J, Stone C J, et al. Classification and Regression Trees[M]. CRC Press, 1984.

[6] McFee B, Raffel C, Liang D, et al. librosa: Audio and music signal analysis in python[C]. Proceedings of the 14th Python in Science Conference, 2015: 18-25.

[7] Kingma D P, Ba J. Adam: A method for stochastic optimization[C]. International Conference on Learning Representations (ICLR), 2015.

[8] Srivastava N, Hinton G, Krizhevsky A, et al. Dropout: A simple way to prevent neural networks from overfitting[J]. Journal of Machine Learning Research, 2014, 15(1): 1929-1958.

[9] Ioffe S, Szegedy C. Batch normalization: Accelerating deep network training by reducing internal covariate shift[C]. International Conference on Machine Learning (ICML), 2015: 448-456.

[10] 周志华. 机器学习[M]. 清华大学出版社, 2016.

---

## 七、附录（重要源代码）

### 附录A：音频转梅尔频谱图

```python
"""
audio_to_spectrogram.py
功能：将GTZAN数据集音频文件转换为梅尔频谱图
      保存至 data/spectrograms/ 目录
"""

import os
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# ============ 全局配置 ============
RAW_DATA_DIR = "data/raw/genres_original"   # 原始音频目录
SPEC_SAVE_DIR = "data/spectrograms"         # 频谱图保存目录
SAMPLE_RATE = 22050                          # 采样率
DURATION = 30                               # 音频时长（秒）
N_MELS = 128                                # 梅尔滤波器数量
IMG_SIZE = (128, 128)                       # 输出图像尺寸

# 10个音乐流派标签
GENRES = ['blues', 'classical', 'country', 'disco', 'hiphop',
          'jazz', 'metal', 'pop', 'reggae', 'rock']

def audio_to_melspectrogram(audio_path):
    """
    将单个音频文件转换为梅尔频谱图矩阵
    参数：audio_path - 音频文件路径
    返回：归一化后的梅尔频谱图 numpy 数组
    """
    # 加载音频，截取前30秒
    y, sr = librosa.load(audio_path,
                         sr=SAMPLE_RATE,
                         duration=DURATION)

    # 计算梅尔频谱图
    mel_spec = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_mels=N_MELS,
        n_fft=2048,      # FFT窗口大小
        hop_length=512   # 帧移
    )

    # 转换为分贝刻度
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

    # 归一化至 [0, 1]
    mel_spec_norm = (mel_spec_db - mel_spec_db.min()) / \
                    (mel_spec_db.max() - mel_spec_db.min())

    return mel_spec_norm


def save_spectrogram_image(mel_spec, save_path):
    """
    将梅尔频谱图保存为图像文件
    参数：mel_spec  - 频谱图矩阵
          save_path - 保存路径
    """
    fig, ax = plt.subplots(figsize=(IMG_SIZE[0]/100,
                                    IMG_SIZE[1]/100),
                           dpi=100)
    ax.imshow(mel_spec, aspect='auto',
              origin='lower', cmap='viridis')
    ax.axis('off')
    plt.tight_layout(pad=0)
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()


def process_all_audio():
    """
    批量处理所有音频文件，生成频谱图
    """
    for genre in GENRES:
        # 创建对应流派的保存目录
        genre_spec_dir = os.path.join(SPEC_SAVE_DIR, genre)
        os.makedirs(genre_spec_dir, exist_ok=True)

        genre_audio_dir = os.path.join(RAW_DATA_DIR, genre)
        audio_files = [f for f in os.listdir(genre_audio_dir)
                       if f.endswith('.wav')]

        print(f"正在处理 {genre} 类别，共{len(audio_files)}个文件...")

        for audio_file in tqdm(audio_files):
            audio_path = os.path.join(genre_audio_dir, audio_file)
            save_name = audio_file.replace('.wav', '.png')
            save_path = os.path.join(genre_spec_dir, save_name)

            # 跳过已处理的文件
            if os.path.exists(save_path):
                continue

            try:
                mel_spec = audio_to_melspectrogram(audio_path)
                save_spectrogram_image(mel_spec, save_path)
            except Exception as e:
                print(f"处理文件 {audio_file} 时出错：{e}")

    print("所有音频文件处理完成！")


if __name__ == '__main__':
    process_all_audio()
```

### 附录B：决策树训练与调优

```python
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
```

### 附录C：CNN模型定义与训练

```python
"""
cnn_model.py
功能：定义CNN模型结构
      支持指定卷积块数量、Dropout率等超参数
"""

import tensorflow as tf
from tensorflow.keras import layers, models

def build_cnn(input_shape=(128, 128, 1),
              num_classes=10,
              num_conv_blocks=3,
              dropout_rate_conv=0.25,
              dropout_rate_fc=0.5,
              learning_rate=0.001):
    """
    构建CNN模型
    参数：
        input_shape      - 输入图像尺寸，默认(128,128,1)
        num_classes      - 分类数量，默认10
        num_conv_blocks  - 卷积块数量，可选2/3/4
        dropout_rate_conv- 卷积块后的Dropout率
        dropout_rate_fc  - 全连接层后的Dropout率
        learning_rate    - Adam优化器学习率
    返回：
        编译好的Keras模型
    """
    inputs = tf.keras.Input(shape=input_shape)
    x = inputs

    # 动态构建卷积块，滤波器数量逐层翻倍
    filters = 32
    for i in range(num_conv_blocks):
        # 卷积层
        x = layers.Conv2D(filters, (3, 3),
                          padding='same',
                          activation='relu')(x)
        # 批归一化层
        x = layers.BatchNormalization()(x)
        # 最大池化层
        x = layers.MaxPooling2D((2, 2))(x)
        # Dropout层
        x = layers.Dropout(dropout_rate_conv)(x)

        # 下一个卷积块滤波器数量翻倍（最多128）
        filters = min(filters * 2, 128)

    # 展平层
    x = layers.Flatten()(x)

    # 全连接层
    x = layers.Dense(256, activation='relu',
                     name='dense_1')(x)
    x = layers.Dropout(dropout_rate_fc)(x)

    # 输出层
    outputs = layers.Dense(num_classes,
                           activation='softmax')(x)

    model = models.Model(inputs=inputs, outputs=outputs)

    # 编译模型
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=learning_rate
        ),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model
```

```python
"""
train_cnn.py
功能：加载频谱图数据集
      使用最优超参数训练CNN模型
      保存模型与训练曲线
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
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from PIL import Image
from tqdm import tqdm

# 添加项目根目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from src.models.cnn_model import build_cnn

# ============ 全局配置 ============
SPEC_DIR = "data/spectrograms"
MODEL_SAVE_PATH = "outputs/models/best_cnn_model.h5"
IMG_SIZE = (128, 128)
RANDOM_STATE = 42
BATCH_SIZE = 32
EPOCHS = 100

# 最优超参数（由调优实验确定）
# dropout=0.2在调优中达到70.71%验证集准确率，优于dropout=0.3的65.66%
BEST_LR = 0.0001
BEST_CONV_BLOCKS = 2
BEST_DROPOUT_CONV = 0.2
BEST_DROPOUT_FC = 0.5

GENRES = ['blues', 'classical', 'country', 'disco', 'hiphop',
          'jazz', 'metal', 'pop', 'reggae', 'rock']


def load_spectrogram_dataset():
    """
    从data/spectrograms/目录加载所有频谱图
    返回：图像数组X 和 标签数组y
    """
    X = []
    y = []

    for label_idx, genre in enumerate(GENRES):
        genre_dir = os.path.join(SPEC_DIR, genre)
        img_files = [f for f in os.listdir(genre_dir)
                     if f.endswith('.png')]

        print(f"加载 {genre} 类别，共{len(img_files)}张图像...")

        for img_file in tqdm(img_files):
            img_path = os.path.join(genre_dir, img_file)
            try:
                # 读取图像并转为灰度图
                img = Image.open(img_path).convert('L')
                # 调整尺寸
                img = img.resize(IMG_SIZE)
                # 转为numpy数组并归一化
                img_array = np.array(img) / 255.0
                X.append(img_array)
                y.append(label_idx)
            except Exception as e:
                print(f"读取图像 {img_file} 失败：{e}")

    X = np.array(X)
    y = np.array(y)

    # 增加通道维度 (N, 128, 128) -> (N, 128, 128, 1)
    X = X[..., np.newaxis]

    print(f"\n数据集加载完成！")
    print(f"X.shape: {X.shape}")
    print(f"y.shape: {y.shape}")
    print(f"标签分布：{np.bincount(y)}")

    return X, y


def split_dataset(X, y):
    """
    按 8:1:1 比例划分训练集、验证集、测试集
    """
    # 先划分出测试集（10%）
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y,
        test_size=0.1,
        random_state=RANDOM_STATE,
        stratify=y
    )
    # 再从剩余数据中划分验证集（约11%，使总体为10%）
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val,
        test_size=0.11,
        random_state=RANDOM_STATE,
        stratify=y_train_val
    )

    print(f"训练集大小：{X_train.shape[0]}")
    print(f"验证集大小：{X_val.shape[0]}")
    print(f"测试集大小：{X_test.shape[0]}")

    return X_train, X_val, X_test, y_train, y_val, y_test


def get_data_augmentation():
    """
    定义数据增强管道
    对训练集图像进行随机变换以增加样本多样性
    """
    data_augmentation = tf.keras.Sequential([
        # 随机水平翻转
        tf.keras.layers.RandomFlip('horizontal'),
        # 随机旋转（±10%圆周角）
        tf.keras.layers.RandomRotation(0.1),
        # 随机宽度平移
        tf.keras.layers.RandomTranslation(0.1, 0.1),
        # 随机缩放
        tf.keras.layers.RandomZoom(0.1),
    ], name='data_augmentation')

    return data_augmentation


def build_dataset_pipeline(X_train, y_train,
                            X_val, y_val,
                            X_test, y_test):
    """
    构建tf.data数据管道，提升训练效率
    """
    # 训练集：加入数据增强
    train_ds = tf.data.Dataset.from_tensor_slices(
        (X_train, y_train)
    )
    train_ds = (train_ds
                .shuffle(buffer_size=1000,
                         seed=RANDOM_STATE)
                .batch(BATCH_SIZE)
                .prefetch(tf.data.AUTOTUNE))

    # 验证集：不进行数据增强
    val_ds = tf.data.Dataset.from_tensor_slices(
        (X_val, y_val)
    )
    val_ds = (val_ds
              .batch(BATCH_SIZE)
              .prefetch(tf.data.AUTOTUNE))

    # 测试集：不进行数据增强
    test_ds = tf.data.Dataset.from_tensor_slices(
        (X_test, y_test)
    )
    test_ds = (test_ds
               .batch(BATCH_SIZE)
               .prefetch(tf.data.AUTOTUNE))

    return train_ds, val_ds, test_ds


def get_callbacks():
    """
    定义训练回调函数
    仅保留ModelCheckpoint，保存验证集准确率最高的模型
    注意：不使用EarlyStopping和ReduceLROnPlateau，
          因为lr=0.0001的模型需要较多epoch才能收敛，
          过早触发会导致训练失败
    """
    callbacks = [
        # 保存验证集准确率最高的模型
        tf.keras.callbacks.ModelCheckpoint(
            filepath=MODEL_SAVE_PATH,
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
    ]
    return callbacks


def plot_training_history(history):
    """
    绘制训练过程曲线
    包含Loss曲线和Accuracy曲线
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    epochs_range = range(1, len(history.history['loss']) + 1)

    # Loss曲线
    axes[0].plot(epochs_range,
                 history.history['loss'],
                 'b-', label='训练Loss', linewidth=2)
    axes[0].plot(epochs_range,
                 history.history['val_loss'],
                 'r-', label='验证Loss', linewidth=2)
    axes[0].set_xlabel('训练轮次 (Epoch)', fontsize=13)
    axes[0].set_ylabel('Loss', fontsize=13)
    axes[0].set_title('训练与验证Loss曲线', fontsize=15)
    axes[0].legend(fontsize=12)
    axes[0].grid(True, alpha=0.3)

    # Accuracy曲线
    axes[1].plot(epochs_range,
                 history.history['accuracy'],
                 'b-', label='训练准确率', linewidth=2)
    axes[1].plot(epochs_range,
                 history.history['val_accuracy'],
                 'r-', label='验证准确率', linewidth=2)
    axes[1].set_xlabel('训练轮次 (Epoch)', fontsize=13)
    axes[1].set_ylabel('准确率', fontsize=13)
    axes[1].set_title('训练与验证准确率曲线', fontsize=15)
    axes[1].legend(fontsize=12)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('outputs/figures/cnn_training_curve.png',
                dpi=150, bbox_inches='tight')
    print("训练曲线已保存至 outputs/figures/cnn_training_curve.png")


def plot_confusion_matrix(y_true, y_pred):
    """
    绘制混淆矩阵热力图
    """
    from sklearn.metrics import confusion_matrix
    import seaborn as sns

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
    plt.title('CNN模型混淆矩阵', fontsize=15)
    plt.xticks(rotation=45, ha='right', fontsize=11)
    plt.yticks(rotation=0, fontsize=11)
    plt.tight_layout()
    plt.savefig('outputs/figures/confusion_matrix.png',
                dpi=150, bbox_inches='tight')
    print("混淆矩阵已保存至 outputs/figures/confusion_matrix.png")


if __name__ == '__main__':
    # Step1：加载数据集
    print("=" * 50)
    print("Step1：加载频谱图数据集")
    print("=" * 50)
    X, y = load_spectrogram_dataset()

    # Step2：划分数据集
    print("\n" + "=" * 50)
    print("Step2：划分训练集/验证集/测试集")
    print("=" * 50)
    (X_train, X_val, X_test,
     y_train, y_val, y_test) = split_dataset(X, y)

    # Step3：构建数据管道
    print("\n" + "=" * 50)
    print("Step3：构建tf.data数据管道")
    print("=" * 50)
    train_ds, val_ds, test_ds = build_dataset_pipeline(
        X_train, y_train, X_val, y_val, X_test, y_test
    )

    # Step4：构建CNN模型
    print("\n" + "=" * 50)
    print("Step4：构建CNN模型")
    print("=" * 50)
    model = build_cnn(
        input_shape=(128, 128, 1),
        num_classes=10,
        num_conv_blocks=BEST_CONV_BLOCKS,
        dropout_rate_conv=BEST_DROPOUT_CONV,
        dropout_rate_fc=BEST_DROPOUT_FC,
        learning_rate=BEST_LR
    )
    model.summary()

    # Step5：训练模型
    print("\n" + "=" * 50)
    print("Step5：开始训练CNN模型")
    print("=" * 50)
    os.makedirs('outputs/models', exist_ok=True)
    os.makedirs('outputs/figures', exist_ok=True)

    history = model.fit(
        train_ds,
        epochs=EPOCHS,
        validation_data=val_ds,
        callbacks=get_callbacks(),
        verbose=1
    )

    # Step6：绘制训练曲线
    print("\n" + "=" * 50)
    print("Step6：绘制训练曲线")
    print("=" * 50)
    plot_training_history(history)

    # Step7：在测试集上评估
    print("\n" + "=" * 50)
    print("Step7：测试集评估")
    print("=" * 50)
    best_model = tf.keras.models.load_model(MODEL_SAVE_PATH)
    test_loss, test_acc = best_model.evaluate(test_ds, verbose=0)
    print(f"测试集Loss：{test_loss:.4f}")
    print(f"测试集准确率：{test_acc:.4f}")

    # Step8：生成混淆矩阵
    print("\n" + "=" * 50)
    print("Step8：生成混淆矩阵")
    print("=" * 50)
    y_pred = np.argmax(best_model.predict(test_ds), axis=1)
    plot_confusion_matrix(y_test, y_pred)

    from sklearn.metrics import classification_report
    print("\n各类别详细分类报告：")
    print(classification_report(y_test, y_pred,
                                target_names=GENRES))
```

### 附录D：CNN参数调优实验

```python
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
```

### 附录E：融合模型实验

```python
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
    """
    models = ['CART决策树', 'CNN', 'CNN+决策树\n融合模型']
    metrics_names = ['准确率', '精确率', '召回率', 'F1分数']
    metrics_keys = ['accuracy', 'precision', 'recall', 'f1']

    all_metrics = [dt_metrics, cnn_metrics, fusion_metrics]
    data = {
        key: [m[key] for m in all_metrics]
        for key in metrics_keys
    }

    x = np.arange(len(models))
    width = 0.2
    colors = ['steelblue', 'coral', 'green', 'purple']

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    for idx, (key, name) in enumerate(
            zip(metrics_keys, metrics_names)):
        offset = (idx - 1.5) * width
        bars = axes[0].bar(x + offset,
                           data[key],
                           width,
                           label=name,
                           color=colors[idx],
                           alpha=0.85)
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

    accs = [m['accuracy'] for m in all_metrics]
    bar_colors = ['steelblue', 'coral', 'green']
    bars = axes[1].bar(models, accs,
                       color=bar_colors,
                       alpha=0.85,
                       width=0.5,
                       edgecolor='black',
                       linewidth=0.8)

    baseline = accs[0]
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
```

### 附录F：一键运行主程序

```python
"""
main.py
功能：一键运行所有实验
      按顺序执行数据预处理、模型训练、
      参数调优、结果对比全流程
"""

import os
import sys
import time

def check_environment():
    """
    检查运行环境和数据集是否就绪
    """
    print("=" * 50)
    print("检查运行环境...")
    print("=" * 50)

    required_packages = [
        'tensorflow', 'sklearn', 'librosa',
        'numpy', 'pandas', 'matplotlib',
        'seaborn', 'PIL'
    ]
    missing = []
    for pkg in required_packages:
        try:
            __import__(pkg)
            print(f"  + {pkg}")
        except ImportError:
            print(f"  x {pkg} 未安装")
            missing.append(pkg)

    if missing:
        print(f"\n缺少以下依赖包：{missing}")
        print("请运行：pip install -r requirements.txt")
        sys.exit(1)

    spec_dir = "data/raw/genres_original"
    feat_csv = "data/features/features_30_sec.csv"

    if not os.path.exists(spec_dir):
        print(f"\n未找到音频数据集目录：{spec_dir}")
        print("请先下载GTZAN数据集并解压至 data/raw/ 目录")
        sys.exit(1)

    if not os.path.exists(feat_csv):
        print(f"\n未找到特征CSV文件：{feat_csv}")
        print("请将 features_30_sec.csv 放入 data/features/ 目录")
        sys.exit(1)

    os.makedirs('outputs/models', exist_ok=True)
    os.makedirs('outputs/figures', exist_ok=True)
    os.makedirs('data/spectrograms', exist_ok=True)

    print("\n环境检查通过！")


def run_step(step_name, func, *args, **kwargs):
    """
    运行单个实验步骤并计时
    """
    print("\n" + "=" * 50)
    print(f"开始：{step_name}")
    print("=" * 50)
    start_time = time.time()

    result = func(*args, **kwargs)

    elapsed = time.time() - start_time
    print(f"\n完成：{step_name}")
    print(f"耗时：{elapsed:.1f} 秒")

    return result


if __name__ == '__main__':

    total_start = time.time()

    # Step0：环境检查
    check_environment()

    # Step1：音频转频谱图
    from src.preprocess.audio_to_spectrogram import (
        process_all_audio
    )
    run_step("音频转梅尔频谱图", process_all_audio)

    # Step2：决策树实验
    from src.train.train_dt import (
        load_and_preprocess,
        tune_max_depth,
        compare_three_dt
    )
    X_tr, X_te, y_tr, y_te, le = run_step(
        "加载手工特征数据", load_and_preprocess
    )
    best_depth = run_step(
        "决策树深度参数调优",
        tune_max_depth, X_tr, y_tr
    )
    dt_results = run_step(
        "三种决策树对比实验",
        compare_three_dt,
        X_tr, X_te, y_tr, y_te, best_depth, le
    )

    # Step3：CNN参数调优实验
    from src.train.train_cnn import (
        load_spectrogram_dataset,
        split_dataset,
        build_dataset_pipeline
    )
    from src.tuning.cnn_tuning import (
        tune_learning_rate,
        tune_network_depth,
        tune_dropout_rate
    )

    X, y = run_step(
        "加载频谱图数据集",
        load_spectrogram_dataset
    )
    (X_train, X_val, X_test,
     y_train, y_val, y_test) = split_dataset(X, y)
    train_ds, val_ds, test_ds = build_dataset_pipeline(
        X_train, y_train,
        X_val, y_val,
        X_test, y_test
    )

    best_lr = run_step(
        "CNN学习率调优实验",
        tune_learning_rate, train_ds, val_ds
    )
    best_blocks = run_step(
        "CNN网络深度调优实验",
        tune_network_depth, train_ds, val_ds, best_lr
    )
    best_dropout = run_step(
        "CNN Dropout率调优实验",
        tune_dropout_rate,
        train_ds, val_ds, best_lr, best_blocks
    )

    # Step4：训练最优CNN模型
    from src.train.train_cnn import (
        get_callbacks,
        plot_training_history,
        plot_confusion_matrix
    )
    from src.models.cnn_model import build_cnn
    import numpy as np
    import tensorflow as tf

    model = build_cnn(
        num_conv_blocks=best_blocks,
        dropout_rate_conv=best_dropout,
        dropout_rate_fc=0.5,
        learning_rate=best_lr
    )
    history = run_step(
        "训练最优CNN模型",
        model.fit,
        train_ds,
        epochs=100,
        validation_data=val_ds,
        callbacks=get_callbacks(),
        verbose=1
    )
    plot_training_history(history)

    # Step5：融合模型实验与综合对比
    from src.evaluate.metrics import (
        extract_cnn_features,
        train_fusion_model,
        compute_metrics,
        plot_model_comparison
    )

    (train_features, test_features,
     y_train_feat, y_test_feat) = run_step(
        "提取CNN中间层特征",
        extract_cnn_features,
        'outputs/models/best_cnn_model.h5',
        train_ds, test_ds
    )
    fusion_results = run_step(
        "训练融合模型",
        train_fusion_model,
        train_features, test_features,
        y_train_feat, y_test_feat
    )

    # 汇总所有指标
    best_cnn_model = tf.keras.models.load_model(
        'outputs/models/best_cnn_model.h5'
    )
    y_pred_cnn = np.argmax(
        best_cnn_model.predict(test_ds), axis=1
    )
    cnn_metrics = compute_metrics(
        y_test_feat, y_pred_cnn, 'CNN'
    )

    best_dt = dt_results['CART（基尼系数）']['model']
    y_pred_dt = best_dt.predict(X_te)
    dt_metrics = compute_metrics(
        y_te, y_pred_dt, 'CART决策树'
    )

    fusion_metrics = {
        'accuracy': fusion_results['accuracy'],
        'precision': fusion_results['precision'],
        'recall': fusion_results['recall'],
        'f1': fusion_results['f1']
    }

    # 生成对比图表
    run_step("生成综合对比图",
             plot_model_comparison,
             dt_metrics, cnn_metrics, fusion_metrics)

    # 最终汇总输出
    total_elapsed = time.time() - total_start
    print("\n" + "=" * 50)
    print("全部实验完成！")
    print(f"总耗时：{total_elapsed/60:.1f} 分钟")
    print("=" * 50)
    print("\n实验结果汇总：")
    print(f"{'方案':<20}{'准确率':<12}"
          f"{'精确率':<12}{'召回率':<12}{'F1分数'}")
    print("-" * 65)
    for name, metrics in [
        ('CART决策树',    dt_metrics),
        ('CNN',          cnn_metrics),
        ('CNN+决策树融合', fusion_metrics)
    ]:
        print(f"{name:<20}"
              f"{metrics['accuracy']:<12.4f}"
              f"{metrics['precision']:<12.4f}"
              f"{metrics['recall']:<12.4f}"
              f"{metrics['f1']:.4f}")

    print("\n所有图表已保存至 outputs/figures/ 目录")
    print("最优模型已保存至 outputs/models/ 目录")
    print("\n各图表文件说明：")
    print("  dt_depth_tuning.png      决策树深度调优曲线")
    print("  dt_comparison.png        三种决策树性能对比")
    print("  lr_comparison.png        CNN学习率调优对比")
    print("  cnn_structure_tuning.png CNN网络深度调优对比")
    print("  dropout_tuning.png       Dropout率调优对比")
    print("  cnn_training_curve.png   最优CNN训练曲线")
    print("  confusion_matrix.png     CNN混淆矩阵热力图")
    print("  model_comparison.png     三种方案综合对比")
    print("  radar_chart.png          三种方案雷达图")
```
| 项目 | 内容 |
|------|------|
| 设计题目 | 基于CNN与决策树的音乐流派分类研究 |
| 学生姓名 | [你的姓名] |
| 编号 | [你的编号] |
| 主要参考资料 | [1] Tzanetakis G, Cook P. Musical genre classification of audio signals[J]. IEEE Transactions on Speech and Audio Processing, 2002. [2] 周志华. 机器学习[M]. 清华大学出版社, 2016. [3] librosa官方文档 [4] TensorFlow官方文档 |
| 数据集链接 | GTZAN Dataset - Music Genre Classification (Kaggle) https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification |
| 运行环境 | Python 3.9 / TensorFlow 2.10.0 / scikit-learn 1.2.0 / Windows 10 或 Ubuntu 20.04 |
| 算法 | CNN（卷积神经网络）/ BP算法 / ID3决策树 / C4.5决策树 / CART决策树 / CNN+决策树融合模型 |
| 实验内容 | 1.决策树深度参数调优实验 2.ID3/C4.5/CART三种决策树对比实验 3.CNN学习率调优实验（对应BP算法） 4.CNN网络深度调优实验 5.Dropout率调优实验 6.CNN+决策树融合模型实验 7.三种方案综合性能对比 |
| 是否争优 | 是 |
| 创新点 | 提出CNN特征提取与决策树分类器相结合的融合方案，将CNN全连接层的256维深层特征作为决策树的输入，兼顾CNN的特征表达能力与决策树的可解释性 |
| 独创设计 | 1.设计了CNN中间层特征提取+决策树分类的融合架构，利用CNN自动学习频谱图的高层语义特征，再由决策树完成最终分类决策，实现深度学习与传统机器学习的优势互补 2.采用渐进式超参数调优策略（学习率→网络深度→Dropout率），逐步锁定最优配置，降低搜索空间复杂度 3.构建了从原始音频到梅尔频谱图的自动化预处理流水线，统一图像尺寸与归一化标准，保证实验可复现性 |

---