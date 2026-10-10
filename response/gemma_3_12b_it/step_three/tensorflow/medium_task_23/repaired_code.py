import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Concatenate, Flatten, Dense, AveragePooling2D

class DL_Model(tf.keras.Model):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Path 1: Single 1x1 convolution
        self.conv2d_path1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu', name='conv2d_path1')

        # Path 2: 1x1 conv + 1x7 + 7x1
        self.conv2d_path2_1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu', name='conv2d_path2_1')
        self.conv2d_path2_2 = Conv2D(filters=64, kernel_size=(1, 7), strides=(1, 1), padding='same', activation='relu', name='conv2d_path2_2')
        self.conv2d_path2_3 = Conv2D(filters=64, kernel_size=(7, 1), strides=(1, 1), padding='same', activation='relu', name='conv2d_path2_3')

        # Path 3: 1x1 conv + (1x7 + 7x1) x 2
        self.conv2d_path3_1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu', name='conv2d_path3_1')
        self.conv2d_path3_2 = Conv2D(filters=64, kernel_size=(1, 7), strides=(1, 1), padding='same', activation='relu', name='conv2d_path3_2')
        self.conv2d_path3_3 = Conv2D(filters=64, kernel_size=(7, 1), strides=(1, 1), padding='same', activation='relu', name='conv2d_path3_3')
        self.conv2d_path3_4 = Conv2D(filters=64, kernel_size=(1, 7), strides=(1, 1), padding='same', activation='relu', name='conv2d_path3_4')
        self.conv2d_path3_5 = Conv2D(filters=64, kernel_size=(7, 1), strides=(1, 1), padding='same', activation='relu', name='conv2d_path3_5')

        # Path 4: Average Pooling + 1x1 conv
        self.average_pooling2d = AveragePooling2D(pool_size=(1, 1), strides=(1, 1), padding='same', name='average_pooling2d')
        self.conv2d_path4 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu', name='conv2d_path4')

        # Fully connected layer
        self.flatten = Flatten(name='flatten')
        self.dense = Dense(units=10, activation='softmax', name='dense')  # CIFAR-10 has 10 classes

    def call(self, inputs):
        # the initial shape of inputs is: (1, 32, 32, 3)
        # Path 1
        x1 = self.conv2d_path1(inputs)  # output tensor shape: (1, 32, 32, 64)

        # Path 2
        x2_1 = self.conv2d_path2_1(inputs)  # output tensor shape: (1, 32, 32, 64)
        x2_2 = self.conv2d_path2_2(x2_1)  # output tensor shape: (1, 32, 32, 64)
        x2_3 = self.conv2d_path2_3(x2_2)  # output tensor shape: (1, 32, 32, 64)
        x2 = x2_3  # output tensor shape: (1, 32, 32, 64)

        # Path 3
        x3_1 = self.conv2d_path3_1(inputs)  # output tensor shape: (1, 32, 32, 64)
        x3_2 = self.conv2d_path3_2(x3_1)  # output tensor shape: (1, 32, 32, 64)
        x3_3 = self.conv2d_path3_3(x3_2)  # output tensor shape: (1, 32, 32, 64)
        x3_4 = self.conv2d_path3_4(x3_3)  # output tensor shape: (1, 32, 32, 64)
        x3_5 = self.conv2d_path3_5(x3_4)  # output tensor shape: (1, 32, 32, 64)
        x3 = x3_5  # output tensor shape: (1, 32, 32, 64)

        # Path 4
        x4 = self.average_pooling2d(inputs)  # output tensor shape of this layer should be (1, 32, 32, 3)
        x4 = self.conv2d_path4(x4)

        # Concatenate paths
        merged = Concatenate(axis=-1)([x1, x2, x3, x4])

        # Flatten and dense layer
        x = self.flatten(merged)
        output = self.dense(x)

        return output


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal((1, 32, 32, 3))  # Example input shape for CIFAR-10
    output_tensor = model(input_tensor)
    print(output_tensor.shape)