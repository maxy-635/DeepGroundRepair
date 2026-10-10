import tensorflow as tf

class DL_Model(tf.keras.Model):
    def __init__(self):
        super(DL_Model, self).__init__()
        # Block 1
        self.conv1_1 = tf.keras.layers.Conv2D(filters=32, kernel_size=1, strides=1, padding='same')
        self.conv1_2 = tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=1, padding='same')
        self.conv1_3 = tf.keras.layers.Conv2D(filters=32, kernel_size=5, strides=1, padding='same')
        self.concat1 = tf.keras.layers.Concatenate(axis=-1)

        # Block 2
        self.branch1 = tf.keras.layers.Conv2D(filters=64, kernel_size=1, strides=1, padding='same')
        self.branch2 = tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=1, padding='same')
        self.branch3 = tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=1, padding='same')
        self.branch4 = tf.keras.layers.Conv2D(filters=64, kernel_size=1, strides=1, padding='same')
        self.branch5 = tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=1, padding='same')
        self.branch6 = tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding='same')
        self.concat2 = tf.keras.layers.Concatenate(axis=-1)

        # Flatten and Dense layers
        self.flatten = tf.keras.layers.Flatten()
        self.dense = tf.keras.layers.Dense(units=10, activation='softmax')  # CIFAR-10 has 10 classes

    def call(self, inputs):
        # Block 1
        x = self.conv1_1(inputs)
        x = self.conv1_2(x)
        x = self.conv1_3(x)
        x = self.concat1([x, x, x])  # Concatenate the three groups

        # Block 2
        x = self.branch1(x)
        x = self.branch2(x)
        x = self.branch3(x)
        x = self.branch4(x)
        x = self.branch5(x)
        x = self.branch6(x)
        x = self.concat2([x, x, x])

        x = self.flatten(x)
        x = self.dense(x)

        return x


if __name__ == "__main__":
    # Define the input shape for CIFAR-10 (32x32x3)
    input_shape = (32, 32, 3)
    input_tensor = tf.random.normal(shape=(1, *input_shape))  # Batch size of 1
    model = DL_Model()
    output_tensor = model(input_tensor)
    print("Output shape:", output_tensor.shape)