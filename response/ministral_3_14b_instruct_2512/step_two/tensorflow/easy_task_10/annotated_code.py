import tensorflow as tf
from tensorflow.keras.layers import Conv2D, DepthwiseConv2D, Flatten, Dense, Activation
from tensorflow.keras.models import Model

class DL_Model(Model):
    def __init__(self):
        super(DL_Model, self).__init__()
        self.conv1x1_1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(2, 2), padding='same', activation='relu')
        self.depthwise_conv = DepthwiseConv2D(kernel_size=(3, 3), strides=(2, 2), padding='same', depth_multiplier=1, activation='relu')
        self.conv1x1_2 = Conv2D(filters=32, kernel_size=(1, 1), strides=(2, 2), padding='same', activation='relu')
        self.flatten = Flatten()
        self.dense = Dense(units=10, activation='softmax')

    def call(self, inputs):
        x = self.conv1x1_1(inputs)
        x = self.depthwise_conv(x)
        x = self.conv1x1_2(x)
        x = self.flatten(x)
        x = self.dense(x)
        return x

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal(shape=(1, 28, 28, 1))  # MNIST images are 28x28 with 1 channel
    output_tensor = model(input_tensor)
    print(output_tensor.shape)