import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPool2D, Add, Dense, Flatten, GlobalAveragePooling2D

class DL_Model(tf.keras.Model):
    def __init__(self):
        super(DL_Model, self).__init__()
        self.conv_block1 = tf.keras.Sequential([
            Conv2D(filters='mask', kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'),
            MaxPool2D(pool_size='mask', strides='mask', padding='mask')
        ])
        self.conv_block2 = tf.keras.Sequential([
            Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'),
            MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid')
        ])
        self.add_layer = Add()
        self.flatten = Flatten()
        self.dense_layer = Dense(10, activation='softmax')

    def call(self, inputs):
        # the initial shape of inputs is: (1, 28, 28, 1)
        x = self.conv_block1(inputs)  # output tensor shape of this layer should be (1, 14, 14, 32)
        y = self.conv_block2(inputs)  # output tensor shape: (1, 14, 14, 64)
        z = self.add_layer([inputs, x, y])
        z = self.flatten(z)
        return self.dense_layer(z)

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal((1, 28, 28, 1))  # MNIST image shape
    output_tensor = model(input_tensor)
    print(output_tensor.shape)