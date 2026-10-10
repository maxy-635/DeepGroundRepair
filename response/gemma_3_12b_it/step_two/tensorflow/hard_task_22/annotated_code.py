import tensorflow as tf
from tensorflow.keras.layers import Conv2D, DepthwiseConv2D, SeparableConv2D, Concatenate, Add, Flatten, Dense

class DL_Model(tf.keras.Model):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Main path
        self.main_path = tf.keras.Sequential(layers='mask')










        # Branch path
        self.branch_path = tf.keras.Sequential([
            tf.keras.layers.Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')
        ])

        # Fusion
        self.fusion = tf.keras.layers.Add()

        # Classification
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(units=128, activation='relu')
        self.dense2 = tf.keras.layers.Dense(units=10, activation='softmax')

    def call(self, inputs):
        # the initial shape of inputs is: (1, 32, 32, 3)
        # Main path
        main_path_output = self.main_path(inputs)

        # Branch path
        branch_path_output = self.branch_path(inputs)

        # Fusion
        fused_output = self.fusion([main_path_output, branch_path_output])

        # Classification
        flattened_output = self.flatten(fused_output)
        dense1_output = self.dense1(flattened_output)
        output = self.dense2(dense1_output)

        return output


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal((1, 32, 32, 3))  # Dummy input tensor (batch_size, height, width, channels)
    output_tensor = model(input_tensor)
    print("Output shape:", output_tensor.shape)