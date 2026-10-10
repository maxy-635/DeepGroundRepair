import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense
from tensorflow.keras.models import Sequential

class DL_Model(tf.keras.Model):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Main path
        self.conv2d_1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', name="conv2d_1")
        self.conv2d_2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', name="conv2d_2")
        self.maxpool2d = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same', name="maxpool2d")

        # Branch path
        self.conv2d_branch = Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu', name="conv2d_branch")

        # Fully connected layers
        self.flatten = Flatten(name="flatten")
        self.dense_1 = Dense(units=128, activation='relu', name="dense_1")
        self.dense_2 = Dense(units=10, activation='softmax', name="dense_2")

    def call(self, inputs):
        # Main path
        main_path = self.conv2d_1(inputs)
        main_path = self.conv2d_2(main_path)
        main_path = self.maxpool2d(main_path)

        # Branch path
        branch_path = self.conv2d_branch(inputs)

        # Combine features
        combined_features = tf.keras.layers.concatenate([main_path, branch_path], axis=-1)
        flattened_features = self.flatten(combined_features)
        dense_1_output = self.dense_1(flattened_features)
        output = self.dense_2(dense_1_output)

        return output


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal((1, 32, 32, 3))  # Example input shape (batch_size, height, width, channels)
    output_tensor = model(input_tensor)
    print("Output shape:", output_tensor.shape)