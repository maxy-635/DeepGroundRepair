import tensorflow as tf
from tensorflow.keras.layers import DepthwiseConv2D, Conv2D, Dropout, Flatten, Dense

class DL_Model(tf.keras.Model):
    def __init__(self):
        super(DL_Model, self).__init__()
        self.depthwise_conv = DepthwiseConv2D(
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='same',
            depth_multiplier=1,
            activation='relu'
        )
        self.dropout1 = Dropout(rate=0.25)
        self.conv2d = Conv2D(
            filters=32,
            kernel_size=(1, 1),
            strides=(1, 1),
            padding='same',
            activation='relu'
        )
        self.dropout2 = Dropout(rate=0.25)
        self.flatten = Flatten()
        self.dense = Dense(
            units=10,
            activation='softmax'
        )

    def call(self, inputs):
        x = self.depthwise_conv(inputs)
        x = self.dropout1(x)
        x = self.conv2d(x)
        x = self.dropout2(x)
        x = self.flatten(x)
        x = self.dense(x)
        return x

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal((1, 28, 28, 1))
    output_tensor = model(input_tensor)
    print(output_tensor.shape)