import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Concatenate, Flatten, Dense
from tensorflow.keras.models import Model

class DL_Model(Model):
    def __init__(self):
        super(DL_Model, self).__init__()
        # Define the three convolutional branches
        self.conv1x1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')
        self.conv3x3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')
        self.conv5x5 = Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')

        # Concatenate the outputs
        self.concatenate = Concatenate(axis=-1)

        # Flatten the concatenated features
        self.flatten = Flatten()

        # Fully connected layers
        self.dense1 = Dense(units=512, activation='relu')
        self.dense2 = Dense(units=10, activation='softmax')  # Assuming CIFAR-10 has 10 classes

    def call(self, inputs):
        # Split the input into three groups along the channel dimension
        input_shape = tf.shape(inputs)
        batch_size = input_shape[0]
        height = input_shape[1]
        width = input_shape[2]
        channels = input_shape[3]

        # Split the input into three equal parts along the channel dimension
        split_channels = channels // 3
        group1 = inputs[:, :, :, :split_channels]
        group2 = inputs[:, :, :, split_channels:2*split_channels]
        group3 = inputs[:, :, :, 2*split_channels:]

        # Apply different convolutional kernels to each group
        conv1x1_out = self.conv1x1(group1)
        conv3x3_out = self.conv3x3(group2)
        conv5x5_out = self.conv5x5(group3)

        # Concatenate the outputs
        concatenated = self.concatenate([conv1x1_out, conv3x3_out, conv5x5_out])

        # Flatten the concatenated features
        flattened = self.flatten(concatenated)

        # Pass through fully connected layers
        dense1_out = self.dense1(flattened)
        output = self.dense2(dense1_out)

        return output

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal(shape=(1, 32, 32, 3))  # Dummy input tensor for CIFAR-10 (32x32x3)
    output_tensor = model(input_tensor)
    print("Input shape:", input_tensor.shape)
    print("Output shape:", output_tensor.shape)