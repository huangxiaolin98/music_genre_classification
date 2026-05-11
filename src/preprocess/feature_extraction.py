"""
feature_extraction.py
功能：从音频文件中提取手工特征（MFCC等）
      用于决策树模型的输入
"""

import os
import numpy as np
import pandas as pd
import librosa
from tqdm import tqdm

# ============ 全局配置 ============
RAW_DATA_DIR = "data/raw/genres_original"
FEATURE_SAVE_PATH = "data/features/features_30_sec.csv"
SAMPLE_RATE = 22050
DURATION = 30

GENRES = ['blues', 'classical', 'country', 'disco', 'hiphop',
          'jazz', 'metal', 'pop', 'reggae', 'rock']


def extract_features(audio_path):
    """
    从单个音频文件中提取特征
    参数：audio_path - 音频文件路径
    返回：特征字典
    """
    y, sr = librosa.load(audio_path, sr=SAMPLE_RATE,
                         duration=DURATION)

    features = {}

    # 过零率
    zcr = librosa.feature.zero_crossing_rate(y)
    features['zcr_mean'] = np.mean(zcr)
    features['zcr_var'] = np.var(zcr)

    # 频谱质心
    spectral_centroid = librosa.feature.spectral_centroid(
        y=y, sr=sr)
    features['spectral_centroid_mean'] = np.mean(spectral_centroid)
    features['spectral_centroid_var'] = np.var(spectral_centroid)

    # 频谱带宽
    spectral_bandwidth = librosa.feature.spectral_bandwidth(
        y=y, sr=sr)
    features['spectral_bandwidth_mean'] = np.mean(spectral_bandwidth)
    features['spectral_bandwidth_var'] = np.var(spectral_bandwidth)

    # 频谱滚降
    spectral_rolloff = librosa.feature.spectral_rolloff(
        y=y, sr=sr)
    features['rolloff_mean'] = np.mean(spectral_rolloff)
    features['rolloff_var'] = np.var(spectral_rolloff)

    # 色度特征
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    features['chroma_stft_mean'] = np.mean(chroma)
    features['chroma_stft_var'] = np.var(chroma)

    # RMS能量
    rms = librosa.feature.rms(y=y)
    features['rms_mean'] = np.mean(rms)
    features['rms_var'] = np.var(rms)

    # 谐波与打击乐分离
    harmony, perceptr = librosa.effects.hpss(y)
    features['harmony_mean'] = np.mean(harmony)
    features['harmony_var'] = np.var(harmony)
    features['perceptr_mean'] = np.mean(perceptr)
    features['perceptr_var'] = np.var(perceptr)

    # 节奏（tempo）
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    features['tempo'] = tempo

    # MFCC（20个系数的均值和方差）
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    for i in range(20):
        features[f'mfcc{i+1}_mean'] = np.mean(mfcc[i])
        features[f'mfcc{i+1}_var'] = np.var(mfcc[i])

    return features


def extract_all_features():
    """
    批量提取所有音频文件的特征并保存为CSV
    """
    all_features = []

    for genre in GENRES:
        genre_dir = os.path.join(RAW_DATA_DIR, genre)
        audio_files = [f for f in os.listdir(genre_dir)
                       if f.endswith('.wav')]

        print(f"正在提取 {genre} 类别特征，"
              f"共{len(audio_files)}个文件...")

        for audio_file in tqdm(audio_files):
            audio_path = os.path.join(genre_dir, audio_file)
            try:
                features = extract_features(audio_path)
                features['filename'] = audio_file
                features['label'] = genre
                features['length'] = DURATION
                all_features.append(features)
            except Exception as e:
                print(f"提取 {audio_file} 特征时出错：{e}")

    # 保存为CSV
    df = pd.DataFrame(all_features)
    os.makedirs(os.path.dirname(FEATURE_SAVE_PATH), exist_ok=True)
    df.to_csv(FEATURE_SAVE_PATH, index=False)
    print(f"\n特征文件已保存至：{FEATURE_SAVE_PATH}")
    print(f"共{len(df)}条样本，{len(df.columns)}列")

    return df


if __name__ == '__main__':
    extract_all_features()
