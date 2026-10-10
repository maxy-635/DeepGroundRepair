import tensorflow as tf
from tensorflow.keras.layers import SeparableConv2D, Concatenate, Conv2D, MaxPool2D, GlobalAveragePooling2D, Dense

class DL_Model(tf.keras.Model):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Block 1: Separable Convolutions with different kernel sizes
        self.sep_conv1_1x1 = SeparableConv2D(
            filters=64,
            kernel_size=(1, 1),
            strides=(1, 1),
            padding='same',
            activation='relu',
            name='sep_conv1_1x1'
        )
        self.sep_conv1_3x3 = SeparableConv2D(
            filters=64,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='same',
            activation='relu',
            name='sep_conv1_3x3'
        )
        self.sep_conv1_5x5 = SeparableConv2D(
            filters=64,
            kernel_size=(5, 5),
            strides=(1, 1),
            padding='same',
            activation='relu',
            name='sep_conv1_5x5'
        )
        self.concat1 = Concatenate(axis=-1, name='mask')

        # Block 2: Multiple Branches
        self.conv2_3x3 = Conv2D(
            filters=128,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='same',
            activation='relu',
            name='conv2_3x3'
        )
        self.conv2_branch1_1x1 = Conv2D(
            filters=64,
            kernel_size=(1, 1),
            strides=(1, 1),
            padding='same',
            activation='relu',
            name='conv2_branch1_1x1'
        )
        self.conv2_branch1_3x3_1 = Conv2D(
            filters=64,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='same',
            activation='relu',
            name='conv2_branch1_3x3_1'
        )
        self.conv2_branch1_3x3_2 = Conv2D(
            filters=64,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='same',
            activation='relu',
            name='conv2_branch1_3x3_2'
        )
        self.max_pool = MaxPool2D(
            pool_size=(2, 2),
            strides=(2, 2),
            padding='valid',
            name='max_pool'
        )
        self.concat2 = Concatenate(axis=-1, name='concat2')

        # Final Layers
        self.global_avg_pool = GlobalAveragePooling2D(name='global_avg_pool')
        self.dense = Dense(
            units=10,
            activation='softmax',
            name='dense'
        )

    def call(self, inputs):
        # the initial shape of inputs is: (1, 32, 32, 3)
        # Block 1
        x = self.sep_conv1_1x1(inputs)  # output tensor shape: (1, 32, 32, 64)
        x = self.sep_conv1_3x3(x)  # output tensor shape: (1, 32, 32, 64)
        x = self.sep_conv1_5x5(x)  # output tensor shape: (1, 32, 32, 64)
        x = self.concat1([x, self.sep_conv1_1x1(inputs), self.sep_conv1_3x3(inputs), self.sep_conv1_5x5(inputs)])

        # Block 2
        branch1 = self.conv2_3x3(inputs)
        branch2 = self.conv2_branch1_1x1(inputs)
        branch2 = self.conv2_branch1_3x3_1(branch2)
        branch2 = self.conv2_branch1_3x3_2(branch2)
        branch3 = self.max_pool(inputs)

        x = self.concat2([branch1, branch2, branch3])

        # Final Layers
        x = self.global_avg_pool(x)
        x = self.dense(x)

        return x


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal((1, 32, 32, 3))  # Example input shape (batch_size, height, width, channels)
    output_tensor = model(input_tensor)
    print("Output shape:", output_tensor.shape)