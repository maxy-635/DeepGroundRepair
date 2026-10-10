import tensorflow as tf
from tensorflow.keras.layers import Conv1D, DepthwiseConv2D, Conv2DTranspose, Conv2D, GlobalAveragePooling2D, Flatten, Dense, MaxPool2D, BatchNormalization, Dropout

class DL_Model(tf.keras.Model):
    def __init__(self):
        super(DL_Model, self).__init__()
        self.conv1 = Conv1D(filters=32, kernel_size=1, strides=2, padding='valid', activation='relu')
        self.depthwise_conv = DepthwiseConv2D(kernel_size=(3, 3), strides=2, padding='valid', depth_multiplier=1, activation='relu')
        self.conv2 = Conv1D(filters=16, kernel_size=1, strides=2, padding='valid', activation='relu')
        self.flatten = Flatten()
        self.dense = Dense(10, activation='softmax')

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.depthwise_conv(x)
        x = self.conv2(x)
        x = self.flatten(x)
        x = self.dense(x)
        return x

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal((1, 28, 28, 1))  # Example input shape
    output_tensor = model(input_tensor)
    print(output_tensor.shape)