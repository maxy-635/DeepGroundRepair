import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPool2D, UpSampling2D, concatenate, Dense, GlobalAveragePooling2D

class DL_Model(tf.keras.Model):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Initial 1x1 convolutional layer
        self.conv1x1_initial = Conv2D(
            filters=64,
            kernel_size=(1, 1),
            strides=(1, 1),
            padding='same',
            activation='relu',
            name='conv1x1_initial'
        )

        # Branch 1: Local features
        self.conv3x3_branch1 = Conv2D(
            filters=64,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='same',
            activation='relu',
            name='conv3x3_branch1'
        )

        # Branch 2: Downsampling, convolution, upsampling
        self.maxpool_branch2 = MaxPool2D(
            pool_size=(2, 2),
            strides=(2, 2),
            padding='same',
            name='maxpool_branch2'
        )
        self.conv3x3_branch2 = Conv2D(
            filters=64,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='same',
            activation='relu',
            name='conv3x3_branch2'
        )
        self.upsampling2D_branch2 = UpSampling2D(
            size=(2, 2),
            name='upsampling2D_branch2'
        )

        # Branch 3: Downsampling, convolution, upsampling
        self.maxpool_branch3 = MaxPool2D(
            pool_size=(2, 2),
            strides=(2, 2),
            padding='same',
            name='maxpool_branch3'
        )
        self.conv3x3_branch3 = Conv2D(
            filters=64,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='same',
            activation='relu',
            name='conv3x3_branch3'
        )
        self.upsampling2D_branch3 = UpSampling2D(
            size=(2, 2),
            name='upsampling2D_branch3'
        )

        # Concatenation and 1x1 convolution
        self.concatenate_layer = concatenate(
            axis=-1,
            name='concatenate_layer'
        )
        self.conv1x1_concat = Conv2D(
            filters=64,
            kernel_size=(1, 1),
            strides=(1, 1),
            padding='same',
            activation='relu',
            name='conv1x1_concat'
        )

        # Fully connected layers
        self.dense1 = Dense(
            units=128,
            activation='relu',
            name='dense1'
        )
        self.dense2 = Dense(
            units=64,
            activation='relu',
            name='dense2'
        )
        self.dense3 = Dense(
            units=10,
            activation='softmax',
            name='dense3'
        )

    def call(self, inputs):
        # Initial convolution
        x = self.conv1x1_initial(inputs)

        # Branch 1
        x1 = self.conv3x3_branch1(x)

        # Branch 2
        x2 = self.maxpool_branch2(x)
        x2 = self.conv3x3_branch2(x2)
        x2 = self.upsampling2D_branch2(x2)

        # Branch 3
        x3 = self.maxpool_branch3(x)
        x3 = self.conv3x3_branch3(x3)
        x3 = self.upsampling2D_branch3(x3)

        # Concatenation
        x = self.concatenate_layer([x1, x2, x3])

        # 1x1 convolution after concatenation
        x = self.conv1x1_concat(x)

        # Fully connected layers
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.dense3(x)

        return x


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal((1, 32, 32, 3))  # Example input shape (batch_size, height, width, channels)
    output_tensor = model(input_tensor)
    print(output_tensor.shape)