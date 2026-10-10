import tensorflow as tf

class DL_Model(tf.keras.Model):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Branch 1: 1x1 Convolution for dimensionality reduction
        self.branch1 = tf.keras.layers.Conv2D(filters=64, kernel_size=1, strides=1, padding='same', activation='relu')

        # Branch 2: 1x1 followed by 3x3 Convolution
        self.branch2_conv1 = tf.keras.layers.Conv2D(filters=64, kernel_size=1, strides=1, padding='same', activation='relu')
        self.branch2_conv2 = tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=1, padding='same', activation='relu')

        # Branch 3: 1x1 followed by 5x5 Convolution
        self.branch3_conv1 = tf.keras.layers.Conv2D(filters=64, kernel_size=1, strides=1, padding='same', activation='relu')
        self.branch3_conv2 = tf.keras.layers.Conv2D(filters=64, kernel_size=5, strides=1, padding='same', activation='relu')

        # Branch 4: 3x3 Max Pooling followed by 1x1 Convolution
        self.branch4_pool = tf.keras.layers.MaxPool2D(pool_size=3, strides=1, padding='same')
        self.branch4_conv = tf.keras.layers.Conv2D(filters=64, kernel_size=1, strides=1, padding='same', activation='relu')

        # Concatenate the outputs of all branches
        self.concatenate = tf.keras.layers.Concatenate(axis=-1)

        # Flatten the features
        self.flatten = tf.keras.layers.Flatten()

        # Fully connected layers for classification
        self.dense1 = tf.keras.layers.Dense(units=128, activation='relu')
        self.dense2 = tf.keras.layers.Dense(units=10, activation='softmax')

    def call(self, inputs):
        # Process each branch
        branch1_output = self.branch1(inputs)
        branch2_output = self.branch2_conv2(self.branch2_conv1(inputs))
        branch3_output = self.branch3_conv2(self.branch3_conv1(inputs))
        branch4_output = self.branch4_conv(self.branch4_pool(inputs))

        # Concatenate the outputs
        merged = self.concatenate([branch1_output, branch2_output, branch3_output, branch4_output])

        # Flatten and pass through fully connected layers
        flattened = self.flatten(merged)
        dense1_output = self.dense1(flattened)
        output = self.dense2(dense1_output)

        return output


if __name__ == "__main__":
    model = DL_Model()
    # CIFAR-10 images are 32x32 with 3 color channels
    input_tensor = tf.random.normal((1, 32, 32, 3))
    output_tensor = model(input_tensor)
    print(output_tensor.shape)