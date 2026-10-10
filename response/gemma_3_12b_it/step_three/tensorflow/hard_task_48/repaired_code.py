import tensorflow as tf
from tensorflow.keras.layers import Conv2D, DepthwiseConv2D, BatchNormalization, Concatenate, AveragePooling2D, Flatten, Dense, Conv2DTranspose, GlobalAveragePooling2D, Reshape

class DL_Model(tf.keras.Model):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Block 1
        self.conv2d_1x1_1 = DepthwiseConv2D(kernel_size=(1, 1), strides=(1, 1), padding='same')
        self.bn_1x1_1 = BatchNormalization(axis=-1)
        self.conv2d_3x3_1 = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same')
        self.bn_3x3_1 = BatchNormalization(axis=-1)
        self.conv2d_5x5_1 = DepthwiseConv2D(kernel_size=(5, 5), strides=(1, 1), padding='same')
        self.bn_5x5_1 = BatchNormalization(axis=-1)

        # Block 2 - Path 1
        self.conv2d_1x1_path1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same')

        # Block 2 - Path 2
        self.avg_pooling_path2 = AveragePooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')
        self.conv2d_1x1_path2 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same')

        # Block 2 - Path 3
        self.conv2d_1x1_path3_1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same')
        self.conv2d_1x3_path3_1 = Conv2D(filters=64, kernel_size=(1, 3), strides=(1, 1), padding='same')
        self.conv2d_3x1_path3_1 = Conv2D(filters=64, kernel_size=(3, 1), strides=(1, 1), padding='same')
        self.concat_path3 = Concatenate(axis=-1)

        # Block 2 - Path 4
        self.conv2d_1x1_path4 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same')
        self.conv2d_3x3_path4 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same')
        self.conv2d_1x3_path4_1 = Conv2D(filters=64, kernel_size=(1, 3), strides=(1, 1), padding='same')
        self.conv2d_3x1_path4_1 = Conv2D(filters=64, kernel_size=(3, 1), strides=(1, 1), padding='same')
        self.concat_path4 = Concatenate(axis=-1)

        # Final Classification
        self.flatten = Flatten()
        self.dense = Dense(units=10, activation='softmax')  # Assuming 10 classes for CIFAR-10

    def call(self, inputs):
        # Block 1
        group1 = self.conv2d_1x1_1(inputs)
        group1 = self.bn_1x1_1(group1)
        group2 = self.conv2d_3x3_1(inputs)
        group2 = self.bn_3x3_1(group2)
        group3 = self.conv2d_5x5_1(inputs)
        group3 = self.bn_5x5_1(group3)

        merged = Concatenate(axis=-1)([group1, group2, group3])

        # Block 2 - Paths
        path1 = self.conv2d_1x1_path1(merged)
        path2 = self.avg_pooling_path2(merged)
        path2 = self.conv2d_1x1_path2(path2)
        path3_1 = self.conv2d_1x1_path3_1(merged)
        path3_1_1 = self.conv2d_1x3_path3_1(path3_1)
        path3_1_2 = self.conv2d_3x1_path3_1(path3_1)
        path3 = self.concat_path3([path3_1_1, path3_1_2])
        path4 = self.conv2d_1x1_path4(merged)
        path4 = self.conv2d_3x3_path4(path4)
        path4_1_1 = self.conv2d_1x3_path4_1(path4)
        path4_1_2 = self.conv2d_3x1_path4_1(path4)
        path4 = self.concat_path4([path4_1_1, path4_1_2])

        combined = Concatenate(axis=-1)([path1, path2, path3, path4])

        # Final Classification
        x = self.flatten(combined)
        output = self.dense(x)

        return output


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal((1, 32, 32, 3))  # Example input shape for CIFAR-10
    output_tensor = model(input_tensor)
    print(output_tensor.shape)