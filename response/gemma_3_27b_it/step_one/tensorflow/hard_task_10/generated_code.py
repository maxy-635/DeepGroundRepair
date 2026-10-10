import tensorflow as tf

class DL_Model(tf.keras.Model):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Feature extraction path 1: 1x1 convolution
        self.conv1x1 = tf.keras.layers.Conv2D(
            filters=32, kernel_size=(1, 1), strides=(1, 1), padding="same", activation="relu"
        )

        # Feature extraction path 2: 1x1, 1x7, 7x1 convolutions
        self.conv1x1_path2 = tf.keras.layers.Conv2D(
            filters=32, kernel_size=(1, 1), strides=(1, 1), padding="same", activation="relu"
        )
        self.conv1x7_path2 = tf.keras.layers.Conv2D(
            filters=32, kernel_size=(1, 7), strides=(1, 1), padding="same", activation="relu"
        )
        self.conv7x1_path2 = tf.keras.layers.Conv2D(
            filters=32, kernel_size=(7, 1), strides=(1, 1), padding="same", activation="relu"
        )

        # Concatenate the outputs of the two paths
        self.concatenate = tf.keras.layers.Concatenate(axis=-1)

        # 1x1 convolution to align dimensions
        self.conv1x1_concat = tf.keras.layers.Conv2D(
            filters=3, kernel_size=(1, 1), strides=(1, 1), padding="same", activation="relu"
        )

        # Branch connecting directly to the input
        self.identity_branch = tf.keras.layers.Identity()

        # Add the main path and the branch
        self.add = tf.keras.layers.Add()

        # Fully connected layers for classification
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(units=128, activation="relu")
        self.dense2 = tf.keras.layers.Dense(units=10, activation="softmax")

    def call(self, inputs):
        # Feature extraction path 1
        path1 = self.conv1x1(inputs)

        # Feature extraction path 2
        path2 = self.conv1x1_path2(inputs)
        path2 = self.conv1x7_path2(path2)
        path2 = self.conv7x1_path2(path2)

        # Concatenate the outputs
        merged = self.concatenate([path1, path2])

        # 1x1 convolution to align dimensions
        merged = self.conv1x1_concat(merged)

        # Add the identity branch
        output = self.add([merged, self.identity_branch(inputs)])

        # Fully connected layers
        output = self.flatten(output)
        output = self.dense1(output)
        output = self.dense2(output)

        return output


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal((1, 32, 32, 3))
    output_tensor = model(input_tensor)
    print(output_tensor.shape)