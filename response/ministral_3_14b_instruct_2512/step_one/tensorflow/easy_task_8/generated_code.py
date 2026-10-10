import tensorflow as tf
from tensorflow.keras.layers import Conv2D, SeparableConv2D, Dropout, Flatten, Dense
from tensorflow.keras.models import Model

class DL_Model(Model):
    def __init__(self):
        super(DL_Model, self).__init__()
        self.depthwise_conv = SeparableConv2D(
            filters=32,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='same',
            activation='relu'
        )
        self.pointwise_conv = Conv2D(
            filters=32,
            kernel_size=(1, 1),
            strides=(1, 1),
            padding='same',
            activation='relu'
        )
        self.dropout1 = Dropout(rate=0.2)
        self.dropout2 = Dropout(rate=0.2)
        self.flatten = Flatten()
        self.dense = Dense(units=10, activation='softmax')

    def call(self, inputs):
        x = self.depthwise_conv(inputs)
        x = self.dropout1(x)
        x = self.pointwise_conv(x)
        x = self.dropout2(x)
        x = self.flatten(x)
        x = self.dense(x)
        return x

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal(shape=(1, 28, 28, 1))  # MNIST images are 28x28 with 1 channel
    output_tensor = model(input_tensor)
    print(output_tensor.shape)