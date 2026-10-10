import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dropout, concatenate, Dense

class DL_Model(tf.keras.Model):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Branch 1: 1x1 convolution, 3x3 convolution
        self.conv2d_1a = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu', name='conv2d_1a')
        self.conv2d_1b = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', name='conv2d_1b')
        self.dropout_1 = Dropout(rate=0.25, name='dropout_1')

        # Branch 2: 1x1 convolution, 1x7 convolution, 7x1 convolution, 3x3 convolution
        self.conv2d_2a = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu', name='conv2d_2a')
        self.conv2d_2b = Conv2D(filters=64, kernel_size=(1, 7), strides=(1, 1), padding='same', activation='relu', name='conv2d_2b')
        self.conv2d_2c = Conv2D(filters=64, kernel_size=(7, 1), strides=(1, 1), padding='same', activation='relu', name='conv2d_2c')
        self.conv2d_2d = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', name='conv2d_2d')
        self.dropout_2 = Dropout(rate=0.25, name='dropout_2')

        # Branch 3: Max Pooling
        self.maxpool2d = MaxPool2D(pool_size='mask', strides='mask', padding='mask', name='maxpool2d')

        # Fully connected layers
        self.dense1 = Dense(units=128, activation='relu', name='dense1')
        self.dense2 = Dense(units=64, activation='relu', name='dense2')
        self.dense3 = Dense(units=10, activation='softmax', name='dense3')  # 10 classes for CIFAR-10

    def call(self, inputs):
        # the initial shape of inputs is: (1, 32, 32, 3)
        # Branch 1
        x1 = self.conv2d_1a(inputs)  # output tensor shape: (1, 32, 32, 64)
        x1 = self.conv2d_1b(x1)  # output tensor shape: (1, 32, 32, 64)
        x1 = self.dropout_1(x1)  # output tensor shape: (1, 32, 32, 64)

        # Branch 2
        x2 = self.conv2d_2a(inputs)  # output tensor shape: (1, 32, 32, 64)
        x2 = self.conv2d_2b(x2)  # output tensor shape: (1, 32, 32, 64)
        x2 = self.conv2d_2c(x2)  # output tensor shape: (1, 32, 32, 64)
        x2 = self.conv2d_2d(x2)  # output tensor shape: (1, 32, 32, 64)
        x2 = self.dropout_2(x2)  # output tensor shape: (1, 32, 32, 64)

        # Branch 3
        x3 = self.maxpool2d(inputs)  # output tensor shape of this layer should be (1, 32, 32, 3)

        # Concatenate the outputs from all branches
        concatenated = concatenate([x1, x2, x3], axis=-1)

        # Fully connected layers
        x = self.dense1(concatenated)
        x = self.dense2(x)
        output = self.dense3(x)

        return output


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal((1, 32, 32, 3))  # Example input shape for CIFAR-10
    output_tensor = model(input_tensor)
    print(output_tensor.shape)