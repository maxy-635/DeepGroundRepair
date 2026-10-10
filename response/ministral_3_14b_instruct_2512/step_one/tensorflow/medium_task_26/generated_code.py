import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Dense, Flatten, Concatenate, Activation, BatchNormalization

class DL_Model(tf.keras.Model):
    def __init__(self):
        super(DL_Model, self).__init__()
        # Compression layer (1x1 convolution)
        self.compress_conv = Conv2D(
            filters=32,
            kernel_size=(1, 1),
            strides=(1, 1),
            padding='same',
            activation='relu'
        )

        # Parallel convolutional layers
        self.conv1x1 = Conv2D(
            filters=64,
            kernel_size=(1, 1),
            strides=(1, 1),
            padding='same',
            activation='relu'
        )

        self.conv3x3 = Conv2D(
            filters=64,
            filters=64,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='same',
            activation='relu'
        )

        # Batch normalization layers
        self.bn1 = BatchNormalization()
        self.bn2 = BatchNormalization()

        # Fully connected layers
        self.flatten = Flatten()
        self.dense1 = Dense(units=128, activation='relu')
        self.dense2 = Dense(units=10, activation='softmax')  # Assuming 10 classes for classification

    def call(self, inputs):
        # Compress input channels
        x = self.compress_conv(inputs)

        # Parallel convolutions
        conv1x1_out = self.conv1x1(x)
        conv3x3_out = self.conv3x3(x)

        # Concatenate outputs
        concat_out = Concatenate(axis=-1)([conv1x1_out, conv3x3_out])

        # Batch normalization
        x = self.bn1(concat_out)
        x = self.bn2(x)

        # Flatten and dense layers
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dense2(x)

        return x

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal(shape=(1, 32, 32, 64))  # Batch size 1, 32x32x64 input
    output_tensor = model(input_tensor)
    print("Output shape:", output_tensor.shape)