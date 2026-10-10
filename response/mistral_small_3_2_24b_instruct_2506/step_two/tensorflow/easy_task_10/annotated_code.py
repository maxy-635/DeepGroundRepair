import tensorflow as tf
from tensorflow.keras.layers import Layer, Conv2D, DepthwiseConv2D, Flatten, Dense

class DL_Model(Layer):
    def __init__(self):
        super(DL_Model, self).__init__()
        self.conv1x1_1 = Conv2D(filters=64, kernel_size=(1, 1), strides=2, padding='same')
        self.depthwise_conv = DepthwiseConv2D(kernel_size=(3, 3), strides=2, padding='same')
        self.conv1x1_2 = Conv2D(filters=32, kernel_size=(1, 1), strides=2, padding='same')
        self.flatten = Flatten()
        self.dense = Dense(units=10, activation='softmax')

    def call(self, inputs):
        x = self.conv1x1_1(inputs)
        x = self.depthwise_conv(x)
        x = self.conv1x1_2(x)
        x = self.flatten(x)
        return self.dense(x)

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal(shape=(1, 28, 28, 1))  # MNIST image shape: (28, 28, 1)
    output_tensor = model(input_tensor)
    print("Input shape:", input_tensor.shape)
    print("Output shape:", output_tensor.shape)