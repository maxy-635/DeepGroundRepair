import tensorflow as tf
from tensorflow.keras.layers import DepthwiseConv2D, LayerNormalization, Conv2D, Flatten, Dense

class DL_Model(tf.keras.Model):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Main path
        self.depthwise_conv = DepthwiseConv2D(
            kernel_size=(7, 7),
            strides=(1, 1),
            padding='same',
            depth_multiplier=1,
            activation='relu'
        )
        self.layer_norm = LayerNormalization(axis=-1, epsilon=1e-6)
        self.conv1 = Conv2D(
            filters=32,
            kernel_size=(1, 1),
            strides=(1, 1),
            padding='same',
            activation='relu'
        )
        self.conv2 = Conv2D(
            filters=32,
            kernel_size=(1, 1),
            strides=(1, 1),
            padding='same',
            activation='relu'
        )

        # Branch path
        self.branch_conv = Conv2D(
            filters=32,
            kernel_size=(1, 1),
            strides=(1, 1),
            padding='same',
            activation='relu'
        )


        # Classification layers
        self.flatten = Flatten()
        self.dense1 = Dense(
            units=128,
            activation='relu'
        )
        self.dense2 = Dense(
            units=10,
            activation='softmax'
        )

    def call(self, inputs):
        # Main path
        main_path = self.depthwise_conv(inputs)
        main_path = self.layer_norm(main_path)
        main_path = self.conv1(main_path)
        main_path = self.conv2(main_path)

        # Branch path
        branch_path = self.branch_conv(inputs)

        # Combine paths
        combined_output = tf.math.add(main_path, branch_path)

        # Classification
        flattened_output = self.flatten(combined_output)
        dense1_output = self.dense1(flattened_output)
        output = self.dense2(dense1_output)

        return output


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal((1, 32, 32, 3))  # Example input: batch size 1, 32x32 image, 3 channels
    output_tensor = model(input_tensor)
    print("Output shape:", output_tensor.shape)