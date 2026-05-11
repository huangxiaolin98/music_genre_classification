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
