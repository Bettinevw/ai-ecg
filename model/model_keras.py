# model_keras.py
# TensorFlow 2.2 compatible Keras model for ECG classification + reconstruction
# Outputs: y_pred (softmax of 2 classes), reconstruction (ECG), bottleneck (encoder GAP vector)

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# ---------- Residual Block (BN -> ReLU -> Conv(stride) -> BN -> ReLU -> Conv(1)) + skip(1x1, stride) ----------
class ResBlock(layers.Layer):
    """
    ResNet-style 1D residual block with downsampling via stride in the first conv.
    """
    def __init__(self, out_ch, kernel_size, stride=2, **kwargs):
        super().__init__(**kwargs)
        self.out_ch = int(out_ch)
        self.kernel_size = int(kernel_size)
        self.stride = int(stride)

        self.bn1 = layers.BatchNormalization()
        self.act1 = layers.ReLU()
        self.conv1 = layers.Conv1D(self.out_ch, self.kernel_size, strides=1, padding="same")

        self.bn2 = layers.BatchNormalization()
        self.act2 = layers.ReLU()
        self.conv2 = layers.Conv1D(self.out_ch, self.kernel_size, strides=self.stride, padding="same")

        self.skip1 = layers.Conv1D(self.out_ch, 1, strides=1, padding="same")
        self.skip2 = layers.MaxPooling1D(pool_size=self.stride, padding="same")
        
    def call(self, x, training=None):
        y = self.bn1(x, training=training)
        y = self.act1(y)
        y = self.conv1(y)
        y = self.bn2(y, training=training)
        y = self.act2(y)
        y = self.conv2(y)
        s = self.skip1(x)
        s = self.skip2(s)
        return layers.add([y, s])
    
    def get_config(self):
        cfg = super().get_config()
        cfg.update({"out_ch": self.out_ch, "kernel_size": self.kernel_size, "stride": self.stride})
        return cfg


# ---------- Model Builder ----------
def build_ecg_hybrid(input_length=2500, n_leads=8, dropout=0.2):
    """
    Build ECG hybrid model.

    Args:
        input_length: number of time samples per ECG (e.g., 2500)
        n_leads: number of leads/channels (e.g., 8 or 12)
        use_block32: include 32-channel residual stage
        use_block64: include 64-channel residual stage
        dropout: dropout rate applied after each stage

    Returns:
        keras.Model with outputs [y_pred, reconstruction, bottleneck]
          - y_pred: (B, 2) softmax
          - reconstruction: (B, input_length, n_leads) tanh
          - bottleneck: (B,) vector after GlobalAveragePooling1D
    """
    inp = keras.Input(shape=(input_length, n_leads), name="ecg")

    # ----- ENCODER -----
    x = layers.Conv1D(16, 1, padding="same", name="initial_conv")(inp)
    
    # print(x.shape)

    # 3x 16-ch blocks (each halves the length approximately with stride=2)
    x = ResBlock(16, kernel_size=7, stride=2, name="b16_r1")(x)
    x = ResBlock(16, kernel_size=7, stride=1, name="b16_r2")(x)
    x = ResBlock(16, kernel_size=7, stride=1, name="b16_r3")(x)
    x = layers.Dropout(dropout, name="drop1")(x)
    
    # print(x.shape)

    x = ResBlock(32, kernel_size=5, stride=2, name="b32_r1")(x)
    x = ResBlock(32, kernel_size=5, stride=1, name="b32_r2")(x)
    x = ResBlock(32, kernel_size=5, stride=1, name="b32_r3")(x)
    x = layers.Dropout(dropout, name="drop2")(x)
    
    # print(x.shape)

    x = ResBlock(64, kernel_size=3, stride=2, name="b64_r1")(x)
    x = ResBlock(64, kernel_size=3, stride=1, name="b64_r2")(x)
    x = ResBlock(64, kernel_size=3, stride=1, name="b64_r3")(x)
    x = layers.Dropout(dropout, name="drop3")(x)
    
    # print(x.shape)
    
    x = layers.Conv1D(128, 1, padding="same", name="final_conv")(x)
    x = layers.BatchNormalization(name="final_batch")(x)
    x = layers.ReLU(name="final_relu")(x)
    # x = layers.Dropout(dropout, name="final_dropout")(x)

    bottleneck = layers.GlobalAveragePooling1D(name="bottleneck")(x)
    bottleneck = layers.Dropout(dropout, name="final_dropout")(bottleneck)
    
    print(bottleneck.shape)

    # Classification head
    # y_pred = layers.Dense(2, activation="softmax", 
    #                         name="y_pred")(bottleneck)

    y_pred = layers.Dense(1, activation="sigmoid", name="y_pred")(bottleneck)
    
    model = keras.Model(
        inputs=inp,
        outputs=[y_pred, bottleneck],
        name="Attia2019"
    )
    
    return model

