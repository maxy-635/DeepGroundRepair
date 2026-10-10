import tensorflow as tf

class DL_Model(tf.keras.Model):
    def __init__(self):
        super(DL_Model, self).__init__()
        # Block 1
        self.conv1 = tf.keras.layers.Conv2D(
            filters=32, kernel_size=3, strides=1, padding='same', activation=tf.keras.activations.relu
        )
        self.conv2 = tf.keras.layers.Conv2D(
            filters=16, kernel_size=3, strides=1, padding='same', activation=tf.keras.activations.relu
        )
        self.add = tf.keras.layers.Add()
        self.block1 = self.add

        # Block 2
        self.pool1 = tf.keras.layers.MaxPool2D(
            pool_size=1, strides=1, padding='valid'
        )
        self.pool2 = tf.keras.layers.MaxPool2D(
            pool_size=2, strides=2, padding='valid'
        )
        self.pool3 = tf.keras.layers.MaxPool2D(
            pool_size=4, strides=4, padding='valid'
        )

        self.flatten = tf.keras.layers.Flatten()
        self.concat = tf.keras.layers.Concatenate(axis=-1)

        # Fully connected layers
        self.dense1 = tf.keras.layers.Dense(
            units=128, activation=tf.keras.activations.relu
        )
        self.dense2 = tf.keras.layers.Dense(
            units=10, activation=tf.keras.activations.softmax  # Assuming 10 classes
        )

    def call(self, inputs):
        # Block 1
        x = self.conv1(inputs)
        x = self.conv2(x)
        main_path = x

        # Branch path
        branch_path = inputs

        # Combine main and branch paths
        combined = self.add([main_path, branch_path])

        # Block 2
        pooled = self.pool1(combined)
        pooled = self.pool2(pooled)
        pooled = self.pool3(pooled)
        flattened = self.flatten(pooled)
        output = self.concat([flattened])

        # Fully connected layers
        x = self.dense1(output)
        return self.dense2(x)


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal(shape=(1, 28, 28, 1))  # Example input shape for MNIST
    output_tensor = model(input_tensor)
    print(output_tensor.shape)