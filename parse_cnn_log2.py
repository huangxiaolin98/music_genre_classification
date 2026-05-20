import re

log_path = '/Users/huangxiaogua/Documents/work/music_genre_classification/cnn_tuning.log'
with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
    lines = f.readlines()

for i, line in enumerate(lines):
    # 找学习率汇总表
    if '学习率调优结果汇总' in line or (i > 5640 and i < 5660 and ('0.1' in line or '0.01' in line or '0.001' in line or '0.0001' in line)):
        print(f"{i}: {line.strip()}")
    # 找最优参数总结
    if '最优学习率' in line and '最优卷积块数' not in line and i < 6000:
        print(f"{i}: {line.strip()}")
    if '最优卷积块数量' in line:
        print(f"{i}: {line.strip()}")
    if '最优Dropout率' in line:
        print(f"{i}: {line.strip()}")
    if '调优实验完成' in line:
        print(f"{i}: {line.strip()}")
    if '最优学习率：' in line and i > 13000:
        print(f"{i}: {line.strip()}")
    if '最优卷积块数' in line and i > 13000:
        print(f"{i}: {line.strip()}")
    if '最终训练准确率' in line and i > 11000:
        print(f"{i}: {line.strip()}")
    if '过拟合差距' in line and i > 11000:
        print(f"{i}: {line.strip()}")

# 也提取每个dropout的 final_train_acc, best_val_acc, overfit_gap
print("\n--- Dropout实验详细数据 ---")
for i, line in enumerate(lines):
    if i > 11000 and ('最终训练准确率' in line or '最优验证准确率' in line or '过拟合差距' in line):
        print(f"{i}: {line.strip()}")
