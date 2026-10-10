import tensorflow as tf
from tensorflow.keras.layers import SeparableConv2D, Conv2D, Concatenate, Add, Flatten, Dense

class DL_Model(tf.keras.Model):
    def __init__(self):
        super(DL_Model, self).__init__()
        # Main path
        self.conv1x1 = SeparableConv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')
        self.conv3x3 = SeparableConv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')
        self.conv5x5 = SeparableConv2D(filters=32, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')

        # Branch path
        self.branch_conv = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')

        # Concatenate and Add layers
        self.concatenate = Concatenate(axis=-1)
        self.add = Add()

        # Fully connected layers
        self.flatten = Flatten()
        self.dense1 = Dense(units=128, activation='relu')
        self.dense2 = Dense(units=10, activation='softmax')

    def call(self, inputs):
        # Main path
        conv1x1_output = self.conv1x1(inputs)
        conv3x3_output = self.conv3x3(inputs)
        conv5x5_output = self.conv5x5(inputs)

        # Concatenate outputs from main path
        main_path_output = self.concatenate([conv1x1_output, conv3x3_output, conv5x5_output])

        # Branch path
        branch_path_output = self.branch_conv(inputs)

        # Add outputs from main and branch paths
        combined_output = self.add([main_path_output, branch_path_output])

        # Fully connected layers
        flattened_output = self.flatten(combined_output)
        output = self.dense1(flattened_output)
        output = self.dense2(output)

        return output

if __name__ == "__main__":
    input_tensor = tf.random.normal(shape=(1, 32, 32, 3))  # Example input shape for CIFAR-10 (batch_size=1, height=32, width=32, channels=3)
    output_tensor = DL_Model()(input_tensor)
    print(output_tensor.shape)