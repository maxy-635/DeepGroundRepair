import tensorflow as tf
from tensorflow.keras.layers import Conv2D, DepthwiseConv2D, SeparableConv2D, BatchNormalization, Concatenate, AveragePooling2D, Dense, Activation, Flatten, Lambda

class DL_Model(tf.keras.Model):
    def __init__(self):
        super(DL_Model, self).__init__()

        # First block: three groups with depthwise separable convolutions of different kernel sizes
        self.conv1x1 = SeparableConv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')
        self.conv3x3 = SeparableConv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')
        self.conv5x5 = SeparableConv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')
        self.bn1 = BatchNormalization(axis=-1)

        # Second block: multiple branches for feature extraction
        # Branch 1: 1x1 conv
        self.branch1_conv1x1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')

        # Branch 2: 1x1 conv -> 1x7 conv -> 7x1 conv -> 3x3 conv
        self.branch2_conv1x1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')
        self.branch2_conv1x7 = Conv2D(filters=64, kernel_size=(1, 7), strides=(1, 1), padding='same', activation='relu')
        self.branch2_conv7x1 = Conv2D(filters=64, kernel_size=(7, 1), strides=(1, 1), padding='same', activation='relu')
        self.branch2_conv3x3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')

        # Branch 3: average pooling
        self.branch3_avg_pool = AveragePooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')

        # Concatenate layers
        self.concat = Concatenate(axis=-1)

        # Fully connected layers
        self.flatten = Flatten()
        self.fc1 = Dense(units=128, activation='relu')
        self.fc2 = Dense(units=10, activation='softmax')

    def call(self, inputs):
        # First block processing
        x1 = self.conv1x1(inputs)
        x2 = self.conv3x3(inputs)
        x3 = self.conv5x5(inputs)
        x = tf.concat([x1, x2, x3], axis=-1)
        x = self.bn1(x)

        # Second block processing
        # Branch 1
        branch1 = self.branch1_conv1x1(x)

        # Branch 2
        branch2 = self.branch2_conv1x1(x)
        branch2 = self.branch2_conv1x7(branch2)
        branch2 = self.branch2_conv7x1(branch2)
        branch2 = self.branch2_conv3x3(branch2)

        # Branch 3
        branch3 = self.branch3_avg_pool(x)

        # Concatenate all branches
        x = self.concat([branch1, branch2, branch3])

        # Fully connected layers
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)

        return x

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal(shape=(1, 32, 32, 3))  # CIFAR-10 image shape: 32x32x3
    output_tensor = model(input_tensor)
    print("Output shape:", output_tensor.shape)