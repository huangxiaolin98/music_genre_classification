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

    # 检查依赖包
    required_packages = [
        'tensorflow', 'sklearn', 'librosa',
        'numpy', 'pandas', 'matplotlib',
        'seaborn', 'PIL'
    ]
    missing = []
    for pkg in required_packages:
        try:
            __import__(pkg)
            print(f"  ✓ {pkg}")
        except ImportError:
            print(f"  ✗ {pkg} 未安装")
            missing.append(pkg)

    if missing:
        print(f"\n缺少以下依赖包：{missing}")
        print("请运行：pip install -r requirements.txt")
        sys.exit(1)

    # 检查数据集目录
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

    # 创建输出目录
    os.makedirs('outputs/models', exist_ok=True)
    os.makedirs('outputs/figures', exist_ok=True)
    os.makedirs('data/spectrograms', exist_ok=True)

    print("\n环境检查通过！")


def run_step(step_name, func, *args, **kwargs):
    """
    运行单个实验步骤并计时
    参数：
        step_name - 步骤名称
        func      - 要运行的函数
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
        plot_model_comparison,
        plot_radar_chart
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
    run_step("生成雷达图",
             plot_radar_chart,
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
