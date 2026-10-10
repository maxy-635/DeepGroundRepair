import tensorflow as tf
from tensorflow.keras.layers import Conv2D, SeparableConv2D, BatchNormalization, Concatenate, AveragePooling2D, Dense, Flatten, Lambda, Activation

class DL_Model(tf.keras.Model):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Block 1: Three parallel separable convolutions with different kernel sizes
        self.sep_conv1x1 = SeparableConv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')
        self.sep_conv3x3 = SeparableConv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')
        self.sep_conv5x5 = SeparableConv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')
        self.bn1 = BatchNormalization(axis=-1)

        # Block 2: Four parallel branches
        # Path 1: 1x1 convolution
        self.conv1_path1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')

        # Path 2: 3x3 average pooling + 1x1 convolution
        self.avg_pool_path2 = AveragePooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')
        self.conv1_path2 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')

        # Path 3: 1x1 convolution + split into 1x3 and 3x1 convolutions
        self.conv1_path3 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')
        self.conv1x3_path3 = Conv2D(filters=32, kernel_size=(1, 3), strides=(1, 1), padding='same', activation='relu')
        self.conv3x1_path3 = Conv2D(filters=32, kernel_size=(3, 1), strides=(1, 1), padding='same', activation='relu')

        # Path 4: 1x1 convolution + 3x3 convolution + split into 1x3 and 3x1 convolutions
        self.conv1_path4 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')
        self.conv3x3_path4 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')
        self.conv1x3_path4 = Conv2D(filters=32, kernel_size=(1, 3), strides=(1, 1), padding='same', activation='relu')
        self.conv3x1_path4 = Conv2D(filters=32, kernel_size=(3, 1), strides=(1, 1), padding='same', activation='relu')

        # Final layers
        self.flatten = Flatten()
        self.dense = Dense(units=10, activation='softmax')

    def call(self, inputs):
        # Block 1 processing
        x1 = self.sep_conv1x1(inputs)
        x2 = self.sep_conv3x3(inputs)
        x3 = self.sep_conv5x5(inputs)
        x_block1 = Concatenate(axis=-1)([x1, x2, x3])
        x_block1 = self.bn1(x_block1)

        # Block 2 processing
        # Path 1
        path1 = self.conv1_path1(x_block1)

        # Path 2
        path2 = self.avg_pool_path2(x_block1)
        path2 = self.conv1_path2(path2)

        # Path 3
        path3 = self.conv1_path3(x_block1)
        path3_1 = self.conv1x3_path3(path3)
        path3_2 = self.conv3x1_path3(path3)
        path3 = Concatenate(axis=-1)([path3_1, path3_2])

        # Path 4
        path4 = self.conv1_path4(x_block1)
        path4 = self.conv3x3_path4(path4)
        path4_1 = self.conv1x3_path4(path4)
        path4_2 = self.conv3x1_path4(path4)
        path4 = Concatenate(axis=-1)([path4_1, path4_2])

        # Concatenate all paths
        x_block2 = Concatenate(axis=-1)([path1, path2, path3, path4])

        # Final layers
        x = self.flatten(x_block2)
        x = self.dense(x)

        return x

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal(shape=(32, 32, 32, 3))  # Batch size 32, 32x32x32 feature map (CIFAR-10 is 32x32x3, but adjusted for demo)
    output_tensor = model(input_tensor)
    print("Output shape:", output_tensor.shape)