import tensorflow as tf
from tensorflow.keras.layers import Conv2D, DepthwiseConv2D, BatchNormalization, Concatenate, AveragePooling2D, Dense, Input
from tensorflow.keras.models import Model

class DL_Model(Model):
    def __init__(self):
        super(DL_Model, self).__init__()
        # First block: Three groups with different kernel sizes
        self.depthwise_conv1 = DepthwiseConv2D(kernel_size=(1, 1), padding='same', activation='relu')
        self.depthwise_conv2 = DepthwiseConv2D(kernel_size=(3, 3), padding='same', activation='relu')
        self.depthwise_conv3 = DepthwiseConv2D(kernel_size=(5, 5), padding='same', activation='relu')
        self.batch_norm1 = BatchNormalization(axis=-1)
        self.batch_norm2 = BatchNormalization(axis=-1)
        self.batch_norm3 = BatchNormalization(axis=-1)
        self.concatenate1 = Concatenate(axis=-1)

        # Second block: Multiple branches for feature extraction
        # Branch 1: 1x1 convolution, 3x3 convolution
        self.conv1x1_1 = Conv2D(filters=64, kernel_size=(1, 1), padding='same', activation='relu')
        self.conv3x3_1 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')

        # Branch 2: 1x1 convolution, 1x7 convolution, 7x1 convolution, 3x3 convolution
        self.conv1x1_2 = Conv2D(filters=64, kernel_size=(1, 1), padding='same', activation='relu')
        self.conv1x7 = Conv2D(filters=64, kernel_size=(1, 7), padding='same', activation='relu')
        self.conv7x1 = Conv2D(filters=64, kernel_size=(7, 1), padding='same', activation='relu')
        self.conv3x3_2 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')

        # Branch 3: Average pooling
        self.avg_pool = AveragePooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')
        self.conv1x1_3 = Conv2D(filters=64, kernel_size=(1, 1), padding='same', activation='relu')

        self.concatenate2 = Concatenate(axis=-1)

        # Final fully connected layers
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = Dense(units=512, activation='relu')
        self.dense2 = Dense(units=10, activation='softmax')

    def call(self, inputs):
        # First block
        x1 = self.depthwise_conv1(inputs)
        x1 = self.batch_norm1(x1)

        x2 = self.depthwise_conv2(inputs)
        x2 = self.batch_norm2(x2)

        x3 = self.depthwise_conv3(inputs)
        x3 = self.batch_norm3(x3)

        x = self.concatenate1([x1, x2, x3])

        # Second block
        # Branch 1
        branch1 = self.conv1x1_1(x)
        branch1 = self.conv3x3_1(branch1)

        # Branch 2
        branch2 = self.conv1x1_2(x)
        branch2 = self.conv1x7(branch2)
        branch2 = self.conv7x1(branch2)
        branch2 = self.conv3x3_2(branch2)

        # Branch 3
        branch3 = self.avg_pool(x)
        branch3 = self.conv1x1_3(branch3)

        x = self.concatenate2([branch1, branch2, branch3])

        # Final fully connected layers
        x = self.flatten(x)
        x = self.dense1(x)
        output = self.dense2(x)

        return output

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.keras.Input(shape=(32, 32, 3))  # CIFAR-10 input shape
    output_tensor = model(input_tensor)
    print(output_tensor.shape)  # Expected output shape: (None, 10)