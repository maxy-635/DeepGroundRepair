import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense, add
from tensorflow.keras.models import Sequential

class DL_Model(tf.keras.Model):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Main path
        self.conv2d_1_main = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')
        self.conv2d_2_main = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')
        self.maxpool2d_1_main = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')
        self.maxpool2d_2_main = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')

        # Branch path
        self.conv2d_1_branch = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')
        self.maxpool2d_1_branch = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')

        # Fully connected layers
        self.flatten = Flatten()
        self.dense_1 = Dense(units=128, activation='relu')
        self.dense_2 = Dense(units=10, activation='softmax')

    def call(self, inputs):
        # Main path
        main_path = self.conv2d_1_main(inputs)
        main_path = self.conv2d_2_main(main_path)
        main_path = self.maxpool2d_1_main(main_path)
        main_path = self.maxpool2d_2_main(main_path)

        # Branch path
        branch_path = self.conv2d_1_branch(inputs)
        branch_path = self.maxpool2d_1_branch(branch_path)

        # Combine paths
        combined_output = add([main_path, branch_path])

        # Fully connected layers
        flattened_output = self.flatten(combined_output)
        dense_1_output = self.dense_1(flattened_output)
        output = self.dense_2(dense_1_output)

        return output


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal((1, 32, 32, 3))  # Dummy input tensor (batch_size, height, width, channels)
    output_tensor = model(input_tensor)
    print("Output shape:", output_tensor.shape)