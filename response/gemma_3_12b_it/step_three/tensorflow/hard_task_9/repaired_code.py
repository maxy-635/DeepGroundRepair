import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Concatenate, Add, Dense

class DL_Model(tf.keras.Model):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Feature extraction branches
        self.branch1 = Conv2D(filters=64, kernel_size=1, strides=1, padding='same', activation='relu', name='branch1_conv1')
        self.branch2 = Conv2D(filters=64, kernel_size=1, strides=1, padding='same', activation='relu', name='branch2_conv1')
        self.branch2 = Conv2D(filters=64, kernel_size=3, strides=1, padding='same', activation='relu', name='branch2_conv2')
        self.branch3 = Conv2D(filters=64, kernel_size=1, strides=1, padding='same', activation='relu', name='branch3_conv1')
        self.branch3 = Conv2D(filters=64, kernel_size=3, strides=1, padding='same', activation='relu', name='branch3_conv2')
        self.branch3 = Conv2D(filters=64, kernel_size=3, strides=1, padding='same', activation='relu', name='branch3_conv3')

        # Concatenate the outputs of the branches
        self.concat = Concatenate(axis=-1, name='concat')

        # 1x1 convolution to adjust the output dimensions
        self.adjust_dim = Conv2D(filters=64, kernel_size=1, strides=1, padding='same', activation=None, name='adjust_dim_conv')

        # Direct connection to the input
        self.input_connection = Conv2D(filters=64, kernel_size=1, strides=1, padding='same', activation='relu', name='input_connection_conv')

        # Fuse the main path and the branch
        self.add = Add(name='add')

        # Fully connected layers
        self.dense1 = Dense(units=128, activation='relu', name='dense1')
        self.dense2 = Dense(units=10, activation='relu', name='dense2')
        self.dense3 = Dense(units=10, activation='softmax', name='dense3')

    def call(self, inputs):
        # Feature extraction branches
        branch1_output = self.branch1(inputs)
        branch2_output = self.branch2(inputs)
        branch3_output = self.branch3(inputs)

        # Concatenate the outputs of the branches
        concatenated = self.concat([branch1_output, branch2_output, branch3_output])

        # 1x1 convolution to adjust the output dimensions
        adjusted_dim = self.adjust_dim(concatenated)

        # Direct connection to the input
        input_connection_output = self.input_connection(inputs)

        # Fuse the main path and the branch
        fused = self.add([adjusted_dim, input_connection_output])

        # Flatten the output for fully connected layers
        flattened = tf.keras.layers.Flatten()(fused)

        # Fully connected layers
        dense1_output = self.dense1(flattened)
        dense2_output = self.dense2(dense1_output)
        output = self.dense3(dense2_output)

        return output


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal((1, 32, 32, 3))  # Example input shape (batch_size, height, width, channels)
    output_tensor = model(input_tensor)
    print("Output shape:", output_tensor.shape)