import tensorflow as tf
from tensorflow.keras.layers import Layer, Conv2D, AveragePooling2D, Concatenate, Flatten, Dense

class DL_Model(Layer):
    def __init__(self):
        super(DL_Model, self).__init__()
        # Path 1: Single 1x1 convolution
        self.conv1x1_path1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same')

        # Path 2: 1x1 convolution followed by 1x7 and 7x1 convolutions
        self.conv1x1_path2 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same')
        self.conv1x7_path2 = Conv2D(filters=64, kernel_size=(1, 7), strides=(1, 1), padding='same')
        self.conv7x1_path2 = Conv2D(filters=64, kernel_size=(7, 1), strides=(1, 1), padding='same')

        # Path 3: 1x1 convolution followed by two sets of 1x7 and 7x1 convolutions
        self.conv1x1_path3 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same')
        self.conv1x7_path3a = Conv2D(filters=64, kernel_size=(1, 7), strides=(1, 1), padding='same')
        self.conv7x1_path3a = Conv2D(filters=64, kernel_size=(7, 1), strides=(1, 1), padding='same')
        self.conv1x7_path3b = Conv2D(filters=64, kernel_size=(1, 7), strides=(1, 1), padding='same')
        self.conv7x1_path3b = Conv2D(filters=64, kernel_size=(7, 1), strides=(1, 1), padding='same')

        # Path 4: Average pooling followed by 1x1 convolution
        self.avg_pool_path4 = AveragePooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')
        self.conv1x1_path4 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same')

        # Concatenation and classification layers
        self.concatenate = Concatenate(axis=-1)
        self.flatten = Flatten()
        self.dense = Dense(units=10, activation='softmax')

    def call(self, inputs):
        # Path 1
        path1 = self.conv1x1_path1(inputs)

        # Path 2
        path2 = self.conv1x1_path2(inputs)
        path2 = self.conv1x7_path2(path2)
        path2 = self.conv7x1_path2(path2)

        # Path 3
        path3 = self.conv1x1_path3(inputs)
        path3 = self.conv1x7_path3a(path3)
        path3 = self.conv7x1_path3a(path3)
        path3 = self.conv1x7_path3b(path3)
        path3 = self.conv7x1_path3b(path3)

        # Path 4
        path4 = self.avg_pool_path4(inputs)
        path4 = self.conv1x1_path4(path4)

        # Concatenate all paths
        concatenated = self.concatenate([path1, path2, path3, path4])

        # Flatten and classify
        flattened = self.flatten(concatenated)
        output = self.dense(flattened)

        return output

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal(shape=(1, 32, 32, 3))  # Dummy input for CIFAR-10
    output_tensor = model(input_tensor)
    print("Output shape:", output_tensor.shape)