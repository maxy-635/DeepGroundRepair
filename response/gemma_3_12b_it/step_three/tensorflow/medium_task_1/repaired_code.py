import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense
from tensorflow.keras.activations import relu, softmax

class DL_Model(tf.keras.Model):
    def __init__(self):
        super(DL_Model, self).__init__()
        self.conv1 = Conv2D(
            filters=32,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='same',
            activation=relu
        )
        self.conv2 = Conv2D(
            filters=64,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='same',
            activation=relu
        )
        self.pool = MaxPool2D(
            pool_size=(2, 2),
            strides=(2, 2),
            padding='valid'
        )
        self.flatten = Flatten()
        self.dense1 = Dense(
            units=128,
            activation=relu
        )
        self.dense2 = Dense(
            units=10,
            activation=softmax
        )

    def call(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.pool(x)
        x = x + tf.identity(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dense2(x)
        return x


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal((1, 32, 32, 3))  # Example input tensor for CIFAR-10
    output_tensor = model(input_tensor)
    print(output_tensor.shape)