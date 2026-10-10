import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, AveragePooling2D, Concatenate, Dense, Flatten, Activation
from tensorflow.keras.models import Model

class DL_Model(Model):
    def __init__(self):
        super(DL_Model, self).__init__()
        # Initial 1x1 convolution
        self.initial_conv = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')

        # Branch 1: Local feature extraction
        self.branch1_conv1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')
        self.branch1_conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')

        # Branch 2: Downsampling and upsampling
        self.branch2_pool = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')
        self.branch2_conv = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')
        self.branch2_upconv = Conv2DTranspose(filters=64, kernel_size=(3, 3), strides=(2, 2), padding='same', activation='relu')

        # Branch 3: Downsampling and upsampling
        self.branch3_pool = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')
        self.branch3_conv = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')
        self.branch3_upconv = Conv2DTranspose(filters=64, kernel_size=(3, 3), strides=(2, 2), padding='same', activation='relu')

        # Concatenate and refine
        self.concat = Concatenate(axis=-1)
        self.refine_conv = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')

        # Fully connected layer
        self.flatten = Flatten()
        self.dense = Dense(units=10, activation='softmax')

    def call(self, inputs):
        # Initial 1x1 convolution
        x = self.initial_conv(inputs)

        # Branch 1
        branch1 = self.branch1_conv1(x)
        branch1 = self.branch1_conv2(branch1)

        # Branch 2
        branch2 = self.branch2_pool(x)
        branch2 = self.branch2_conv(branch2)
        branch2 = self.branch2_upconv(branch2)

        # Branch 3
        branch3 = self.branch3_pool(x)
        branch3 = self.branch3_conv(branch3)
        branch3 = self.branch3_upconv(branch3)

        # Concatenate and refine
        x = self.concat([branch1, branch2, branch3])
        x = self.refine_conv(x)

        # Fully connected layer
        x = self.flatten(x)
        x = self.dense(x)

        return x

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal((32, 32, 32, 3))  # Example input shape for CIFAR-10 (batch_size, height, width, channels)
    output_tensor = model(input_tensor)
    print(output_tensor.shape)