import tensorflow as tf
from tensorflow.keras.layers import AveragePooling2D, Flatten, Concatenate, Dense, Reshape, Conv2D

class DL_Model(tf.keras.Model):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Block 1
        self.avg_pool_1x1 = AveragePooling2D(pool_size=(1, 1), strides=(1, 1), padding='valid')
        self.avg_pool_2x2 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')
        self.avg_pool_4x4 = AveragePooling2D(pool_size=(4, 4), strides=(4, 4), padding='valid')
        self.flatten_1 = Flatten()
        self.concat_1 = Concatenate(axis=-1)
        self.dense_1 = Dense(128, activation='relu')
        self.reshape_1 = Reshape((1, 1, 128))

        # Block 2
        self.conv2d_1_1_branch1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='valid', activation='relu')
        self.conv2d_3_3_branch1 = Conv2D(filters='mask', kernel_size='mask', strides='mask', padding='mask', activation='mask')

        self.conv2d_1_1_branch2 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='valid', activation='relu')
        self.conv2d_1_7_branch2 = Conv2D(filters=32, kernel_size=(1, 7), strides=(1, 1), padding='valid', activation='relu')
        self.conv2d_7_1_branch2 = Conv2D(filters=32, kernel_size=(7, 1), strides=(1, 1), padding='valid', activation='relu')
        self.conv2d_3_3_branch2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu')
        self.avg_pool_branch2 = tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')
        self.concat_2 = Concatenate(axis=-1)
        self.dense_2 = Dense(64, activation='relu')
        self.dense_3 = Dense(10, activation='softmax')  # Assuming 10 classes for MNIST

    def call(self, inputs):
        # the initial shape of inputs is: (1, 28, 28, 1)
        # Block 1
        x = self.avg_pool_1x1(inputs)  # output tensor shape: (1, 28, 28, 1)
        x = self.flatten_1(x)  # output tensor shape: (1, 784)
        y = self.avg_pool_2x2(inputs)  # output tensor shape: (1, 14, 14, 1)
        y = self.flatten_1(y)  # output tensor shape: (1, 196)
        z = self.avg_pool_4x4(inputs)  # output tensor shape: (1, 7, 7, 1)
        z = self.flatten_1(z)  # output tensor shape: (1, 49)
        combined = self.concat_1([x, y, z])  # output tensor shape: (1, 1029)
        dense_output = self.dense_1(combined)  # output tensor shape: (1, 128)
        reshaped_output = self.reshape_1(dense_output)  # output tensor shape: (1, 1, 1, 128)

        # Block 2
        branch1 = self.conv2d_1_1_branch1(reshaped_output)  # output tensor shape: (1, 1, 1, 32)
        branch1 = self.conv2d_3_3_branch1(branch1)
        branch2 = self.conv2d_1_1_branch2(reshaped_output)
        branch2 = self.conv2d_1_7_branch2(branch2)
        branch2 = self.conv2d_7_1_branch2(branch2)
        branch2 = self.conv2d_3_3_branch2(branch2)
        branch2 = self.avg_pool_branch2(branch2)
        combined_branches = self.concat_2([branch1, branch2])
        dense_output_2 = self.dense_2(combined_branches)
        output = self.dense_3(dense_output_2)

        return output


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal((1, 28, 28, 1))  # Example input shape for MNIST (batch_size, height, width, channels)
    output_tensor = model(input_tensor)
    print("Output shape:", output_tensor.shape)