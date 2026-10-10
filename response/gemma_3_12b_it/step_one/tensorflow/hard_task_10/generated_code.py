import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Concatenate, Add, Dense

class DL_Model(tf.keras.Model):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Feature extraction path 1: 1x1 convolution
        self.conv1x1_path1 = Conv2D(
            filters=64,
            kernel_size=(1, 1),
            strides=(1, 1),
            padding='same',
            activation='relu',
            name='conv1x1_path1'
        )

        # Feature extraction path 2: sequence of convolutions (1x1, 1x7, 7x1)
        self.conv1x1_path2 = Conv2D(
            filters=64,
            kernel_size=(1, 1),
            strides=(1, 1),
            padding='same',
            activation='relu',
            name='conv1x1_path2'
        )
        self.conv1x7_path2 = Conv2D(
            filters=64,
            kernel_size=(1, 7),
            strides=(1, 1),
            padding='same',
            activation='relu',
            name='conv1x7_path2'
        )
        self.conv7x1_path2 = Conv2D(
            filters=64,
            kernel_size=(7, 1),
            strides=(1, 1),
            padding='same',
            activation='relu',
            name='conv7x1_path2'
        )

        # Concatenate the outputs of the two paths
        self.concatenate = Concatenate(
            axis=-1,
            name='concatenate'
        )

        # 1x1 convolution to align the output dimensions
        self.conv1x1_align = Conv2D(
            filters=64,
            kernel_size=(1, 1),
            strides=(1, 1),
            padding='same',
            activation='relu',
            name='conv1x1_align'
        )

        # Direct connection to the input
        self.direct_connection = Conv2D(
            filters=64,
            kernel_size=(1, 1),
            strides=(1, 1),
            padding='same',
            activation='relu',
            name='direct_connection'
        )

        # Merge the outputs of the main path and the branch
        self.add = Add(
            name='add'
        )

        # Fully connected layers
        self.dense1 = Dense(
            units=128,
            activation='relu',
            name='dense1'
        )
        self.dense2 = Dense(
            units=10,
            activation='softmax',
            name='dense2'
        )

    def call(self, inputs):
        # Feature extraction path 1
        x1 = self.conv1x1_path1(inputs)

        # Feature extraction path 2
        x2 = self.conv1x1_path2(inputs)
        x2 = self.conv1x7_path2(x2)
        x2 = self.conv7x1_path2(x2)

        # Concatenate the outputs of the two paths
        merged = self.concatenate([x1, x2])

        # 1x1 convolution to align the output dimensions
        aligned = self.conv1x1_align(merged)

        # Direct connection to the input
        branch = self.direct_connection(inputs)

        # Merge the outputs of the main path and the branch
        merged_path = self.add([aligned, branch])

        # Flatten the output
        flattened = tf.keras.layers.Flatten()(merged_path)

        # Fully connected layers
        x = self.dense1(flattened)
        output = self.dense2(x)

        return output


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal((1, 32, 32, 3))  # Dummy input tensor (batch_size, height, width, channels)
    output_tensor = model(input_tensor)
    print("Output shape:", output_tensor.shape)