import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Concatenate, Add, Flatten, Dense, Input

class DL_Model(tf.keras.Model):
    def __init__(self):
        super(DL_Model, self).__init__()

        self.input_tensor = Input(shape=(32, 32, 3))

        # Main Path
        self.split = tf.split(self.input_tensor, num_or_size_splits=3, axis=-1)
        self.conv2d_1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')
        self.conv2d_2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')

        # Branch Path
        self.conv2d_branch = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')

        # Fusion
        self.concatenate = Concatenate(axis=-1)
        self.add = Add()
        self.flatten = Flatten()
        self.dense = Dense(units=10, activation='softmax')

    def call(self, inputs):
        # the initial shape of inputs is: (1, 32, 32, 3)
        # Main Path
        group1 = self.split[0]  # output tensor shape: (None, 32, 32, 1)
        group2 = self.split[1]  # output tensor shape: (None, 32, 32, 1)
        group3 = self.split[2]  # output tensor shape: (None, 32, 32, 1)

        processed_group2 = self.conv2d_1(group2)  # output tensor shape: (None, 32, 32, 32)
        combined_group = tf.concat([group3, processed_group2], axis=-1)  # output tensor shape: (None, 32, 32, 33)
        main_path_output = self.conv2d_2(combined_group)  # output tensor shape: (None, 32, 32, 64)
        
        # Branch Path
        branch_path_output = self.conv2d_branch(inputs)  # output tensor shape of this layer should be (1, 32, 32, 64)

        # Fusion
        fused_output = self.add([main_path_output, branch_path_output])
        flattened_output = self.flatten(fused_output)
        output = self.dense(flattened_output)

        return output


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal((1, 32, 32, 3))  # Dummy input tensor
    output_tensor = model(input_tensor)
    print(output_tensor.shape)