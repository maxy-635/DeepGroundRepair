import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Concatenate, Flatten, Dense

class DL_Model(tf.keras.Model):
    def __init__(self):
        super(DL_Model, self).__init__()

        # 1x1 convolutional layer to compress input channels
        self.conv1x1 = Conv2D(
            filters=64,
            kernel_size=(1, 1),
            strides=(1, 1),
            padding='same',
            activation='relu',
            name='conv1x1'
        )

        # Parallel convolutional layers
        self.conv1x1_parallel = Conv2D(
            filters=32,
            kernel_size=(1, 1),
            strides=(1, 1),
            padding='same',
            activation='relu',
            name='conv1x1_parallel'
        )

        self.conv3x3_parallel = Conv2D(
            filters=32,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='same',
            activation='relu',
            name='conv3x3_parallel'
        )

        # Concatenate the results of parallel layers
        self.concatenate = Concatenate(
            axis=-1,
            name='concatenate'
        )

        # Flatten the feature map
        self.flatten = Flatten(
            name='flatten'
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
        # Compress input channels
        x = self.conv1x1(inputs)

        # Expand features through parallel layers
        x1 = self.conv1x1_parallel(x)
        x2 = self.conv3x3_parallel(x)

        # Concatenate the results
        x = self.concatenate([x1, x2])

        # Flatten the feature map
        x = self.flatten(x)

        # Pass through fully connected layers
        x = self.dense1(x)
        output = self.dense2(x)

        return output


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal((1, 32, 32, 64))  # Dummy input tensor
    output_tensor = model(input_tensor)
    print(output_tensor.shape)