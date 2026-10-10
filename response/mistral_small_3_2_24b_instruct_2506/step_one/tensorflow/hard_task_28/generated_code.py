import tensorflow as tf
from tensorflow.keras.layers import DepthwiseConv2D, LayerNormalization, Conv2D, Add, Flatten, Dense

class DL_Model(tf.keras.Model):
    def __init__(self):
        super(DL_Model, self).__init__()
        # Main path
        self.depthwise_conv = DepthwiseConv2D(kernel_size=7, strides=1, padding='same', depth_multiplier=1, activation='relu')
        self.layer_norm = LayerNormalization(axis=-1, epsilon=1e-6, center=True, scale=True)
        self.conv1 = Conv2D(filters=64, kernel_size=1, strides=1, padding='same', activation='relu')
        self.conv2 = Conv2D(filters=64, kernel_size=1, strides=1, padding='same', activation='relu')

        # Branch path (identity connection)
        self.add = Add()

        # Classification head
        self.flatten = Flatten()
        self.dense1 = Dense(units=128, activation='relu')
        self.dense2 = Dense(units=10, activation='softmax')

    def call(self, inputs):
        # Main path
        x = self.depthwise_conv(inputs)
        x = self.layer_norm(x)
        x = self.conv1(x)
        x = self.conv2(x)

        # Branch path
        branch = inputs

        # Combine paths
        x = self.add([x, branch])

        # Classification head
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dense2(x)

        return x

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal(shape=(1, 32, 32, 3))  # CIFAR-10 input shape
    output_tensor = model(input_tensor)
    print("Model output shape:", output_tensor.shape)