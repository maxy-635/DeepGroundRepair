import tensorflow as tf
from tensorflow.keras.layers import Layer, Conv2D, AveragePooling2D, Conv2DTranspose, Dense, Concatenate

class DL_Model(tf.keras.Model):
    def __init__(self):
        super(DL_Model, self).__init__()
        # Initial 1x1 convolution
        self.initial_conv = Conv2D(filters=64, kernel_size=1, padding='same')

        # First branch: local feature extraction
        self.branch1_conv1 = Conv2D(filters=64, kernel_size=3, padding='same')
        self.branch1_conv2 = Conv2D(filters=64, kernel_size=3, padding='same')

        # Second branch: downsampling and upsampling
        self.branch2_pool = AveragePooling2D(pool_size=2, strides=2, padding='valid')
        self.branch2_conv = Conv2D(filters=64, kernel_size=3, padding='same')
        self.branch2_upconv = Conv2DTranspose(filters=64, kernel_size=3, strides=2, padding='valid')

        # Third branch: downsampling and upsampling
        self.branch3_pool = AveragePooling2D(pool_size=2, strides=2, padding='valid')
        self.branch3_conv = Conv2D(filters=64, kernel_size=3, padding='same')
        self.branch3_upconv = Conv2DTranspose(filters=64, kernel_size=3, strides=2, padding='valid')

        # Concatenation and final layers
        self.concat = Concatenate(axis=-1)
        self.final_conv = Conv2D(filters=64, kernel_size=1, padding='same')
        self.flatten = tf.keras.layers.Flatten()
        self.dense = Dense(units=10, activation='softmax')

    def call(self, inputs):
        # Initial 1x1 convolution
        x = self.initial_conv(inputs)

        # First branch
        branch1 = self.branch1_conv1(x)
        branch1 = self.branch1_conv2(branch1)

        # Second branch
        branch2 = self.branch2_pool(x)
        branch2 = self.branch2_conv(branch2)
        branch2 = self.branch2_upconv(branch2)

        # Third branch
        branch3 = self.branch3_pool(x)
        branch3 = self.branch3_conv(branch3)
        branch3 = self.branch3_upconv(branch3)

        # Concatenate branches
        concatenated = self.concat([branch1, branch2, branch3])

        # Final layers
        x = self.final_conv(concatenated)
        x = self.flatten(x)
        output = self.dense(x)

        return output

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal(shape=(1, 32, 32, 3))  # Dummy input tensor for CIFAR-10
    output_tensor = model(input_tensor)
    print("Model created successfully. Input shape:", input_tensor.shape, "Output shape:", output_tensor.shape)