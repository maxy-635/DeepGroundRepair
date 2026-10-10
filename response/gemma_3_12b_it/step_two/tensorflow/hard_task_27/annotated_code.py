import tensorflow as tf
from tensorflow.keras.layers import DepthwiseConv2D, LayerNormalization, Dense, Add

class DL_Model(tf.keras.Model):
    def __init__(self):
        super(DL_Model, self).__init__()
        self.depthwise_conv = DepthwiseConv2D(
            kernel_size=(7, 7),
            strides=1,
            padding='same',
            depth_multiplier=1,
            activation='relu'
        )
        self.layer_norm = LayerNormalization(axis=-1, epsilon=1e-6)
        self.dense1 = Dense(
            units=3072,
            activation='relu'
        )
        self.dense2 = Dense(
            units=3072,
            activation='relu'
        )
        self.dense3 = Dense(
            units=10,
            activation='softmax'
        )

    def call(self, inputs):
        x = self.depthwise_conv(inputs)
        x = self.layer_norm(x)
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.dense3(x)
        return x

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal((1, 32, 32, 3))
    output_tensor = model(input_tensor)
    print(output_tensor.shape)