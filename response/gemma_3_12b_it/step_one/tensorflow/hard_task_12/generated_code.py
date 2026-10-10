import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Concatenate, Add, Flatten, Dense, Input

class DL_Model(tf.keras.Model):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Main path
        self.conv2d_1x1_main = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu', name="conv2d_1x1_main")
        self.conv2d_1x1_main2 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu', name="conv2d_1x1_main2")
        self.conv2d_3x3_main = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', name="conv2d_3x3_main")

        # Branch path
        self.conv2d_3x3_branch = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', name="conv2d_3x3_branch")

        # Fully connected layers
        self.flatten = Flatten(name="flatten")
        self.dense1 = Dense(units=128, activation='relu', name="dense1")
        self.dense2 = Dense(units=10, activation='softmax', name="dense2")  # Assuming 10 classes

    def call(self, inputs):
        # Main path
        main_path = self.conv2d_1x1_main(inputs)
        main_path_1 = self.conv2d_1x1_main2(main_path)
        main_path_2 = self.conv2d_3x3_main(main_path)
        main_path_combined = Concatenate(axis=-1)([main_path_1, main_path_2])

        # Branch path
        branch_path = self.conv2d_3x3_branch(inputs)

        # Combine main and branch paths
        combined_path = Add()([main_path_combined, branch_path])

        # Classification
        flattened = self.flatten(combined_path)
        dense1_output = self.dense1(flattened)
        output = self.dense2(dense1_output)

        return output


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal(shape=(1, 32, 32, 64))  # Dummy input tensor
    output_tensor = model(input_tensor)
    print("Output shape:", output_tensor.shape)