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
