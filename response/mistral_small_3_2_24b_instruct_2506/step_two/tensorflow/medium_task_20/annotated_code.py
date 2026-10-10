import tensorflow as tf
from tensorflow.keras.layers import Layer, Conv2D, MaxPool2D, Concatenate, Flatten, Dense
from tensorflow.keras.models import Model

class DL_Model(Model):
    def __init__(self):
        super(DL_Model, self).__init__()
        # Path 1: 1x1 convolution
        self.conv1x1_1 = Conv2D(filters=64, kernel_size=(1, 1), padding='same', activation='relu')

        # Path 2: 1x1 convolution followed by two 3x3 convolutions
        self.conv1x1_2 = Conv2D(filters=64, kernel_size=(1, 1), padding='same', activation='relu')
        self.conv3x3_1 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')
        self.conv3x3_2 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')

        # Path 3: 1x1 convolution followed by a single 3x3 convolution
        self.conv1x1_3 = Conv2D(filters=64, kernel_size=(1, 1), padding='same', activation='relu')
        self.conv3x3_3 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')

        # Path 4: Max pooling followed by 1x1 convolution
        self.max_pool = MaxPool2D(pool_size=(3, 3), strides=(1, 1), padding='same')
        self.conv1x1_4 = Conv2D(filters=64, kernel_size=(1, 1), padding='same', activation='relu')

        # Concatenate and flatten
        self.concatenate = Concatenate(axis=-1)
        self.flatten = Flatten()

        # Dense layers
        self.dense = Dense(units=128, activation='relu')
        self.output_layer = Dense(units=10, activation='softmax')

    def call(self, inputs):
        # Path 1
        path1 = self.conv1x1_1(inputs)

        # Path 2
        path2 = self.conv1x1_2(inputs)
        path2 = self.conv3x3_1(path2)
        path2 = self.conv3x3_2(path2)

        # Path 3
        path3 = self.conv1x1_3(inputs)
        path3 = self.conv3x3_3(path3)

        # Path 4
        path4 = self.max_pool(inputs)
        path4 = self.conv1x1_4(path4)

        # Concatenate all paths
        concatenated = self.concatenate([path1, path2, path3, path4])

        # Flatten and pass through dense layers
        flattened = self.flatten(concatenated)
        dense_output = self.dense(flattened)
        output = self.output_layer(dense_output)

        return output

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal(shape=(1, 32, 32, 3))  # CIFAR-10 image shape
    output_tensor = model(input_tensor)
    print("Input shape:", input_tensor.shape)
    print("Output shape:", output_tensor.shape)