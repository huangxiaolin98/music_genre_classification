"""
font_config.py
统一配置matplotlib中文字体
解决中文显示为方框的问题
"""

import matplotlib.pyplot as plt
import matplotlib

# 设置中文字体（按优先级尝试）
font_list = ['PingFang SC', 'Heiti SC', 'STHeiti', 'Arial Unicode MS']

for font_name in font_list:
    try:
        matplotlib.rcParams['font.sans-serif'] = [font_name]
        matplotlib.rcParams['axes.unicode_minus'] = False
        break
    except Exception:
        continue

# 确保负号正常显示
matplotlib.rcParams['axes.unicode_minus'] = False
