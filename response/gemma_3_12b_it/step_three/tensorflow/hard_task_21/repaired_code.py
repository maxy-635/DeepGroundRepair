import tensorflow as tf
from tensorflow.keras.layers import Lambda, DepthwiseConv2D, Conv2D, Concatenate, Add, Dense, Flatten

class DL_Model(tf.keras.Model):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Main Path
        self.split_channels = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))
        self.depthwise_conv1x1 = DepthwiseConv2D(kernel_size=(1, 1), strides=(1, 1), padding='same', depth_multiplier=1, activation='relu')
        self.depthwise_conv3x3 = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', depth_multiplier=1, activation='relu')
        self.depthwise_conv5x5 = DepthwiseConv2D(kernel_size=(5, 5), strides=(1, 1), padding='same', depth_multiplier=1, activation='relu')
        self.concatenate = Concatenate(axis=-1)

        # Branch Path
        self.conv1x1 = Conv2D(filters=3, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')

        # Fusion
        self.add = Add()

        # Classification Head
        self.flatten = Flatten()
        self.dense1 = Dense(units=128, activation='relu')
        self.dense2 = Dense(units=10, activation='softmax')

    def call(self, inputs):
        # the initial shape of inputs is: (1, 32, 32, 3)
        # Main Path
        split_outputs = self.split_channels(inputs)  # output tensor shape: (unknown)
        conv1x1_output = self.depthwise_conv1x1(split_outputs[0])  # output tensor shape: (1, 32, 32, 1)
        conv3x3_output = self.depthwise_conv3x3(split_outputs[1])  # output tensor shape: (1, 32, 32, 1)
        conv5x5_output = self.depthwise_conv5x5(split_outputs[2])  # output tensor shape: (1, 32, 32, 1)
        main_path_output = self.concatenate([conv1x1_output, conv3x3_output, conv5x5_output])  # output tensor shape: (1, 32, 32, 3)

        # Branch Path
        branch_path_output = self.conv1x1(inputs)  # output tensor shape of this layer should be (1, 32, 32, 3)

        # Fusion
        fusion_output = self.add([main_path_output, branch_path_output])

        # Classification Head
        flatten_output = self.flatten(fusion_output)
        dense1_output = self.dense1(flatten_output)
        output = self.dense2(dense1_output)

        return output


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal((1, 32, 32, 3))  # Example input tensor (batch_size, height, width, channels)
    output_tensor = model(input_tensor)
    print("Output shape:", output_tensor.shape)