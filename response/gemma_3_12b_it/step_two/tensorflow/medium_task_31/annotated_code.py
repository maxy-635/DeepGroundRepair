import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Concatenate, Flatten, Dense

class DL_Model(tf.keras.Model):
    def __init__(self):
        super(DL_Model, self).__init__()
        # Convolutional layers with different kernel sizes
        self.conv1x1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu', name='conv1x1')
        self.conv3x3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', name='conv3x3')
        self.conv5x5 = Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu', name='conv5x5')

        # Concatenation layer
        self.concat = Concatenate(axis=-1, name='concat')

        # Flatten layer
        self.flatten = Flatten(name='flatten')

        # Fully connected layers
        self.dense1 = Dense(units=128, activation='relu', name='dense1')
        self.dense2 = Dense(units=10, activation='softmax', name='dense2')  # 10 classes for CIFAR-10

    def call(self, inputs):
        # Split the input along the channel dimension (not necessary in this case as Conv2D expects a 4D tensor)
        # group1 = inputs[:, :, :, :32]  # Assuming 32 channels
        # group2 = inputs[:, :, :, 32:64]
        # group3 = inputs[:, :, :, 64:]

        # Apply different convolutional kernels to each group
        x1 = self.conv1x1(inputs)
        x2 = self.conv3x3(inputs)
        x3 = self.conv5x5(inputs)

        # Concatenate the outputs
        merged = self.concat([x1, x2, x3])

        # Flatten the features
        flattened = self.flatten(merged)

        # Pass through fully connected layers
        dense1_output = self.dense1(flattened)
        output = self.dense2(dense1_output)

        return output


if __name__ == "__main__":
    model = DL_Model()
    # Dummy input tensor for validation (CIFAR-10: 32x32x3)
    input_tensor = tf.random.normal((1, 32, 32, 3))  # Batch size of 1
    output_tensor = model(input_tensor)
    print("Output shape:", output_tensor.shape)  # Expected: (1, 10)