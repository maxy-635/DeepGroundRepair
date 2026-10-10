import tensorflow as tf
from tensorflow.keras.layers import Conv2D, DepthwiseConv2D, Flatten, Dense, Activation, BatchNormalization

class DL_Model(tf.keras.Model):
    def __init__(self):
        super(DL_Model, self).__init__()

        self.conv1x1_1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(2, 2), padding='same', activation='relu', name='conv1x1_1')
        self.depthwise_conv = DepthwiseConv2D(kernel_size=(3, 3), strides=(2, 2), padding='same', depth_multiplier=1, activation='relu', name='depthwise_conv')
        self.conv1x1_2 = Conv2D(filters=16, kernel_size=(1, 1), strides=(2, 2), padding='same', activation='relu', name='conv1x1_2')
        self.flatten = Flatten(name='flatten')
        self.dense = Dense(units=10, activation='softmax', name='dense')

    def call(self, inputs):
        x = self.conv1x1_1(inputs)
        x = self.depthwise_conv(x)
        x = self.conv1x1_2(x)
        x = self.flatten(x)
        x = self.dense(x)
        return x


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal((1, 28, 28, 1))  # Dummy input tensor for MNIST (batch_size, height, width, channels)
    output_tensor = model(input_tensor)
    print(output_tensor.shape)