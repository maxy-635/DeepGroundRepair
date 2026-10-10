import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPool2D, Concatenate, Flatten, Dense, Lambda, Activation, BatchNormalization
from tensorflow.keras.models import Model

class DL_Model(Model):
    def __init__(self):
        super(DL_Model, self).__init__()
        # Branch 1: 1x1 convolution
        self.branch1_conv1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')

        # Branch 2: 1x1 conv followed by 3x3 conv
        self.branch2_conv1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')
        self.branch2_conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')

        # Branch 3: 1x1 conv followed by 5x5 conv
        self.branch3_conv1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')
        self.branch3_conv2 = Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')

        # Branch 4: 3x3 max pooling followed by 1x1 conv
        self.branch4_pool = MaxPool2D(pool_size=(3, 3), strides=(1, 1), padding='same')
        self.branch4_conv = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')

        # Fully connected layers
        self.flatten = Flatten()
        self.dense1 = Dense(units=128, activation='relu')
        self.dense2 = Dense(units=10, activation='softmax')

    def call(self, inputs):
        # Branch 1
        branch1 = self.branch1_conv1(inputs)

        # Branch 2
        branch2 = self.branch2_conv2(self.branch2_conv1(inputs))

        # Branch 3
        branch3 = self.branch3_conv2(self.branch3_conv1(inputs))

        # Branch 4
        branch4 = self.branch4_conv(self.branch4_pool(inputs))

        # Concatenate all branches
        concatenated = Concatenate(axis=-1)([branch1, branch2, branch3, branch4])

        # Flatten and fully connected layers
        flattened = self.flatten(concatenated)
        dense1 = self.dense1(flattened)
        outputs = self.dense2(dense1)

        return outputs

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal(shape=(1, 32, 32, 3))  # CIFAR-10 image shape: 32x32x3
    output_tensor = model(input_tensor)
    print(output_tensor.shape)