import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Add, Dense, Flatten, MultiHeadAttention

class DL_Model(tf.keras.Model):
    def __init__(self):
        super(DL_Model, self).__init__()
        # Branch 1: 1x1 convolution
        self.branch1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='valid', activation='relu')

        # Branch 2: 1x1 convolution followed by 3x3 convolution
        self.branch2 = tf.keras.Sequential([
            Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='valid', activation='relu'),
            Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')
        ])

        # Branch 3: 1x1 convolution followed by two 3x3 convolutions
        self.branch3 = tf.keras.Sequential([
            Conv2D(filters=64, filters=64, kernel_size=(1, 1), strides=(1, 1), padding='valid', activation='relu'),
            Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'),
            Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')
        ])

        # Concatenate the outputs of the three branches
        self.concat = tf.keras.layers.Concatenate(axis=3)

        # 1x1 convolution to adjust the output dimensions
        self.final_conv = Conv2D(filters=3, kernel_size=(1, 1), strides=(1, 1), padding='valid', activation='relu')

        # Fully connected layers for classification
        self.dense1 = Dense(units=128, activation='relu')
        self.dense2 = Dense(units=64, activation='relu')
        self.dense3 = Dense(units=10, activation='relu')

    def call(self, inputs):
        # Branch 1 output
        branch1_output = self.branch1(inputs)

        # Branch 2 output
        branch2_output = self.branch2(inputs)

        # Branch 3 output
        branch3_output = self.branch3(inputs)

        # Concatenate the outputs
        concatenated_output = self.concat([branch1_output, branch2_output, branch3_output])

        # Adjust dimensions with 1x1 convolution
        adjusted_output = self.final_conv(concatenated_output)

        # Add the branch directly connected to the input
        main_path = inputs + adjusted_output

        # Flatten and pass through dense layers
        flattened = Flatten()(main_path)
        output = self.dense1(flattened)
        output = self.dense2(output)
        output = self.dense3(output)

        return output

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal(shape=(1, 32, 32, 3))  # Example input shape for CIFAR-10
    output_tensor = model(input_tensor)
    print(output_tensor.shape)