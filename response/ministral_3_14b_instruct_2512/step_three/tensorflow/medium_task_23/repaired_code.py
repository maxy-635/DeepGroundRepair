import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Dense, Flatten, Concatenate, AveragePooling2D, BatchNormalization, ReLU
from tensorflow.keras.models import Model

class DL_Model(Model):
    def __init__(self):
        super(DL_Model, self).__init__()
        # Path 1: Single 1x1 convolution
        self.path1_conv = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')

        # Path 2: 1x1 followed by 1x7 and 7x1 convolutions
        self.path2_conv1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')
        self.path2_conv2 = Conv2D(filters=64, kernel_size=(1, 7), strides=(1, 1), padding='same', activation='relu')
        self.path2_conv3 = Conv2D(filters=64, kernel_size=(7, 1), strides=(1, 1), padding='same', activation='relu')

        # Path 3: 1x1 followed by two sets of 1x7 and 7x1 convolutions
        self.path3_conv1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')
        self.path3_conv2 = Conv2D(filters=64, kernel_size=(1, 7), strides=(1, 1), padding='same', activation='relu')
        self.path3_conv3 = Conv2D(filters=64, kernel_size=(7, 1), strides=(1, 1), padding='same', activation='relu')
        self.path3_conv4 = Conv2D(filters=64, kernel_size=(1, 7), strides=(1, 1), padding='same', activation='relu')
        self.path3_conv5 = Conv2D(filters=64, kernel_size=(7, 1), strides=(1, 1), padding='same', activation='relu')

        # Path 4: Average pooling followed by 1x1 convolution
        self.path4_pool = AveragePooling2D(pool_size=(2, 2), strides=(1, 1), padding='same')
        self.path4_conv = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')

        # Final layers
        self.flatten = Flatten()
        self.dense = Dense(units=10, activation='softmax')

    def call(self, inputs):
        # Path 1
        path1 = self.path1_conv(inputs)

        # Path 2
        path2 = self.path2_conv1(inputs)
        path2 = self.path2_conv2(path2)
        path2 = self.path2_conv3(path2)

        # Path 3
        path3 = self.path3_conv1(inputs)
        path3 = self.path3_conv2(path3)
        path3 = self.path3_conv3(path3)
        path3 = self.path3_conv4(path3)
        path3 = self.path3_conv5(path3)

        # Path 4
        path4 = self.path4_pool(inputs)
        path4 = self.path4_conv(path4)

        # Concatenate all paths
        concatenated = Concatenate(axis=-1)([path1, path2, path3, path4])

        # Final layers
        x = self.flatten(concatenated)
        output = self.dense(x)

        return output

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal(shape=(1, 32, 32, 3))  # CIFAR-10 image shape: 32x32x3
    output_tensor = model(input_tensor)
    print("Input shape:", input_tensor.shape)
    print("Output shape:", output_tensor.shape)