# parse_cnn_log.py
# 作用：一次性日志诊断脚本
#       从 cnn_tuning.log 中提取 val_accuracy 记录，
#       并定位各轮超参数实验（学习率/卷积块数量/Dropout率）的分界线，
#       用于快速排查调优日志中的关键结果，不参与主训练流程。

import re

log_path = '/Users/huangxiaogua/Documents/work/music_genre_classification/cnn_tuning.log'
with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
    lines = f.readlines()

# 提取所有包含 val_accuracy 的行
val_lines = []
for i, line in enumerate(lines):
    if 'val_accuracy:' in line:
        val_lines.append((i, line.strip()))

print(f"总共找到 {len(val_lines)} 个 val_accuracy 记录")

# 打印前20个和最后20个
print("\n--- 前20个 val_accuracy ---")
for i, (ln, line) in enumerate(val_lines[:20]):
    print(f"{ln}: {line[-120:]}")

print("\n--- 最后20个 val_accuracy ---")
for i, (ln, line) in enumerate(val_lines[-20:]):
    print(f"{ln}: {line[-120:]}")

# 尝试定位实验分界线
print("\n--- 查找实验分界线 ---")
for i, line in enumerate(lines):
    if '当前学习率' in line or '当前卷积块数量' in line or '当前Dropout率' in line:
        print(f"{i}: {line.strip()}")
    if '最终验证准确率' in line or '最优验证准确率' in line or '参数量' in line:
        print(f"{i}: {line.strip()}")
