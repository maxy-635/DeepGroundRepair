import tensorflow as tf
from tensorflow.keras.layers import MaxPool2D, Flatten, Concatenate, Dense, Reshape, Conv2D, Input
from tensorflow.keras.models import Model

class DL_Model(Model):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Block 1
        self.maxpool1 = MaxPool2D(pool_size=(1, 1), strides=(1, 1), padding='valid')
        self.maxpool2 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid')
        self.maxpool3 = MaxPool2D(pool_size=(4, 4), strides=(4, 4), padding='valid')
        self.flatten1 = Flatten()
        self.flatten2 = Flatten()
        self.flatten3 = Flatten()
        self.concat1 = Concatenate(axis=-1)
        self.dense1 = Dense(128, activation='relu')
        self.reshape = Reshape((1, 1, 128))

        # Block 2
        self.conv1x1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')
        self.conv3x3_1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')
        self.conv5x5 = Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')
        self.maxpool4 = MaxPool2D(pool_size=(1, 1), strides=(1, 1), padding='valid')
        self.concat2 = Concatenate(axis=-1)
        self.flatten4 = Flatten()
        self.dense2 = Dense(10, activation='softmax')

    def call(self, inputs):
        # the initial shape of inputs is: (1, 28, 28, 1)
        # Block 1
        x1 = self.maxpool1(inputs)  # output tensor shape: (1, 28, 28, 1)
        x2 = self.maxpool2(inputs)  # output tensor shape: (1, 14, 14, 1)
        x3 = self.maxpool3(inputs)  # output tensor shape: (1, 7, 7, 1)

        x1_flat = self.flatten1(x1)  # output tensor shape: (1, 784)
        x2_flat = self.flatten2(x2)  # output tensor shape: (1, 196)
        x3_flat = self.flatten3(x3)  # output tensor shape: (1, 49)

        x = self.concat1([x1_flat, x2_flat, x3_flat])  # output tensor shape: (1, 1029)
        x = self.dense1(x)  # output tensor shape: (1, 128)
        x = self.reshape(x)  # output tensor shape: (1, 1, 1, 128)

        # Block 2
        x1 = self.conv1x1(x)  # output tensor shape: (1, 1, 1, 64)
        x2 = self.conv3x3_1(x)  # output tensor shape: (1, 1, 1, 64)
        x3 = self.conv5x5(x)  # output tensor shape: (1, 1, 1, 64)
        x4 = self.maxpool4(x)  # output tensor shape of this layer should be (1, 1, 1, 128)

        x = self.concat2([x1, x2, x3, x4])
        x = self.flatten4(x)
        x = self.dense2(x)

        return x


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal((1, 28, 28, 1))  # Dummy input tensor for MNIST (batch_size, height, width, channels)
    output_tensor = model(input_tensor)
    print(output_tensor.shape)