import tensorflow as tf
from tensorflow.keras.layers import Input, SeparableConv2D, Concatenate, Dense, Flatten
from tensorflow.keras.models import Model

class DL_Model(Model):
    def __init__(self):
        super(DL_Model, self).__init__()
        # Channel group 1: 1x1 separable convolution
        self.sep_conv1x1 = SeparableConv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')
        # Channel group 2: 3x3 separable convolution
        self.sep_conv3x3 = SeparableConv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')
        # Channel group 3: 5x5 separable convolution
        self.sep_conv5x5 = SeparableConv2D(filters=32, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')
        # Concatenate the outputs
        self.concat = Concatenate(axis=-1)
        # Flatten the concatenated features
        self.flatten = Flatten()
        # Fully connected layers
        self.dense1 = Dense(units=1024, activation='relu')
        self.dense2 = Dense(units=512, activation='relu')
        self.dense3 = Dense(units=10, activation='softmax')

    def call(self, inputs):
        # Split input into three channel groups (assuming input has 3 channels)
        input_shape = tf.shape(inputs)
        batch_size = input_shape[0]
        height = input_shape[1]
        width = input_shape[2]
        channels = input_shape[3]

        # Split into three groups (simplified approach - in practice you might need to reshape)
        group1 = inputs[:, :, :, 0:1]  # First channel
        group2 = inputs[:, :, :, 1:2]  # Second channel
        group3 = inputs[:, :, :, 2:3]  # Third channel

        # Process each group
        out1 = self.sep_conv1x1(group1)
        out2 = self.sep_conv3x3(group2)
        out3 = self.sep_conv5x5(group3)

        # Concatenate the outputs
        concatenated = self.concat([out1, out2, out3])

        # Pass through fully connected layers
        x = self.flatten(concatenated)
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.dense3(x)

        return x

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.keras.Input(shape=(32, 32, 3))
    output_tensor = model(input_tensor)
    model = Model(inputs=input_tensor, outputs=output_tensor)
    model.summary()