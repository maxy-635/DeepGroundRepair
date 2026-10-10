import tensorflow as tf
from tensorflow.keras.layers import Layer, Conv2D, MaxPool2D, UpSampling2D, Dense, Flatten
from tensorflow.experimental import numpy as tnp

class DL_Model(tf.keras.Model):
    def __init__(self):
        super(DL_Model, self).__init__()
        # Initial 1x1 convolutional layer
        self.initial_conv = Conv2D(filters=64, kernel_size=(1, 1), padding='same')

        # Branch 1: 3x3 convolutional layer
        self.branch1_conv = Conv2D(filters=64, kernel_size=(3, 3), padding='same')

        # Branch 2: Max pooling + 3x3 convolution + upsampling
        self.branch2_pool = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')
        self.branch2_conv = Conv2D(filters=64, kernel_size=(3, 3), padding='same')
        self.branch2_upsample = UpSampling2D(size=(2, 2))

        # Branch 3: Max pooling + 3x3 convolution + upsampling
        self.branch3_pool = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')
        self.branch3_conv = Conv2D(filters=64, kernel_size=(3, 3), padding='same')
        self.branch3_upsample = UpSampling2D(size=(2, 2))

        # Fusion layer
        self.fusion_conv = Conv2D(filters=192, kernel_size=(1, 1), padding='same')

        # Fully connected layers
        self.flatten = Flatten()
        self.fc1 = Dense(units=512, activation='relu')
        self.fc2 = Dense(units=256, activation='relu')
        self.fc3 = Dense(units=10, activation='softmax')

    def call(self, inputs):
        # Initial convolution
        x = self.initial_conv(inputs)

        # Branch 1
        branch1 = self.branch1_conv(x)

        # Branch 2
        branch2 = self.branch2_pool(x)
        branch2 = self.branch2_conv(branch2)
        branch2 = self.branch2_upsample(branch2)

        # Branch 3
        branch3 = self.branch3_pool(x)
        branch3 = self.branch3_conv(branch3)
        branch3 = self.branch3_upsample(branch3)

        # Concatenate branches
        concatenated = tnp.concatenate([branch1, branch2, branch3], axis=-1)

        # Fusion convolution
        fused = self.fusion_conv(concatenated)

        # Fully connected layers
        x = self.flatten(fused)
        x = self.fc1(x)
        x = self.fc2(x)
        output = self.fc3(x)

        return output

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal(shape=(1, 32, 32, 3))  # CIFAR-10 input shape
    output_tensor = model(input_tensor)
    print("Output shape:", output_tensor.shape)