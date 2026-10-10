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

        # Feature extraction path 2: 1x1, 1x3, and 3x1 convolutions
        self.conv1x1_path2 = Conv2D(
            filters=64,
            kernel_size=(1, 1),
            strides=(1, 1),
            padding='same',
            activation='relu',
            name='conv1x1_path2'
        )
        self.conv1x3_path2 = Conv2D(
            filters=64,
            kernel_size=(1, 3),
            strides=(1, 1),
            padding='same',
            activation='relu',
            name='conv1x3_path2'
        )
        self.conv3x1_path2 = Conv2D(
            filters=64,
            kernel_size=(3, 1),
            strides=(1, 1),
            padding='same',
            activation='relu',
            name='conv3x1_path2'
        )

        # Concatenate the outputs from both paths
        self.concatenate = Concatenate(
            axis=-1,
            name='concatenate'
        )

        # 1x1 convolution after concatenation
        self.conv1x1_output = Conv2D(filters='mask', kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu', name='conv1x1_output')








        # Direct connection from input
        self.add = Add(
            name='add'
        )

        # Fully connected layers for classification
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
        # the initial shape of inputs is: (1, 32, 32, 3)
        # Feature extraction path 1
        x = self.conv1x1_path1(inputs)  # output tensor shape: (1, 32, 32, 64)

        # Feature extraction path 2
        x2 = self.conv1x1_path2(inputs)  # output tensor shape: (1, 32, 32, 64)
        x2 = self.conv1x3_path2(x2)  # output tensor shape: (1, 32, 32, 64)
        x2 = self.conv3x1_path2(x2)  # output tensor shape: (1, 32, 32, 64)

        # Concatenate the outputs
        x = self.concatenate([x, x2])  # output tensor shape: (1, 32, 32, 128)

        # 1x1 convolution after concatenation
        x = self.conv1x1_output(x)  # output tensor shape of this layer should be (1, 32, 32, 3)

        # Add the direct connection
        x = self.add([inputs, x])

        # Fully connected layers
        x = self.dense1(x)
        x = self.dense2(x)

        return x


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal((1, 32, 32, 3))  # Dummy input tensor with shape (batch_size, height, width, channels)
    output_tensor = model(input_tensor)
    print("Output shape:", output_tensor.shape)