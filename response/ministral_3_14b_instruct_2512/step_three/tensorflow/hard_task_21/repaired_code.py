import tensorflow as tf
from tensorflow.keras.layers import Conv2D, DepthwiseConv2D, SeparableConv2D, Concatenate, Add, Dense, Lambda, Reshape, Flatten, Activation
from tensorflow.keras.models import Model

class DL_Model(Model):
    def __init__(self):
        super(DL_Model, self).__init__()
        # Main path - three groups with different kernel sizes
        self.sep_conv_1x1 = SeparableConv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')
        self.sep_conv_3x3 = SeparableConv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')
        self.sep_conv_5x5 = SeparableConv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')

        # Branch path - 1x1 convolution
        self.branch_conv = Conv2D(filters=192, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')

        # Final layers
        self.flatten = Flatten()
        self.dense1 = Dense(units=128, activation='relu')
        self.dense2 = Dense(units=10, activation='softmax')

    def call(self, inputs):
        # the initial shape of inputs is: (1, 32, 32, 3)
        # Split input into three groups along channel dimension
        input_shape = tf.shape(inputs)  # output tensor shape: (4)
        batch_size = input_shape[0]  # output tensor shape: (1)
        channels = input_shape[-1]  # output tensor shape: (1)

        # Split channels into three equal parts (assuming divisible by 3)
        split_channels = channels // 3  # output tensor shape: (1)
        group1 = inputs[:, :, :, :split_channels]  # output tensor shape: (1, 32, 32, 1)
        group2 = inputs[:, :, :, split_channels:2*split_channels]  # output tensor shape: (1, 32, 32, 1)
        group3 = inputs[:, :, :, 2*split_channels:]  # output tensor shape: (1, 32, 32, 1)

        # Main path processing
        main1 = self.sep_conv_1x1(group1)  # output tensor shape: (1, 32, 32, 64)
        main2 = self.sep_conv_3x3(group2)  # output tensor shape: (1, 32, 32, 64)
        main3 = self.sep_conv_5x5(group3)  # output tensor shape: (1, 32, 32, 64)

        # Concatenate main path outputs
        main_output = Concatenate(axis=-1)([main1, main2, main3])  # output tensor shape: (1, 32, 32, 192)

        # Branch path processing
        branch_output = self.branch_conv(inputs)  # output tensor shape of this layer should be (1, 32, 32, 192)

        # Add main and branch outputs
        combined = Add()([main_output, branch_output])

        # Final classification layers
        x = self.flatten(combined)
        x = self.dense1(x)
        x = self.dense2(x)

        return x

if __name__ == "__main__":
    model = DL_Model()
    # Create a dummy input tensor with shape (batch_size, height, width, channels)
    # CIFAR-10 images are 32x32 with 3 channels
    input_tensor = tf.random.normal(shape=(1, 32, 32, 3))
    output_tensor = model(input_tensor)
    print("Output shape:", output_tensor.shape)