import tensorflow as tf

class DL_Model(tf.keras.Model):
    def __init__(self):
        super(DL_Model, self).__init__()
        # Block 1: Initial convolutional layer
        self.conv1 = tf.keras.layers.Conv1D(filters=32, kernel_size=3, strides=1, padding='same')

        # Block 1: Split into two groups along the last dimension
        self.split = tf.keras.layers.Lambda(lambda x: tf.split(x, num_or_size_splits=2, axis=-1))

        # Group 1: 1x1 convolution, depthwise separable convolution, and another 1x1 convolution
        self.conv1_1x1 = tf.keras.layers.Conv1D(filters=32, kernel_size=1, strides=1, padding='same')
        self.depthwise_conv = tf.keras.layers.DepthwiseConv2D(kernel_size=(1, 1), strides=1, padding='same', depth_multiplier=1, activation=None)
        self.conv1_1x1_after = tf.keras.layers.Conv1D(filters=32, kernel_size=1, strides=1, padding='same')

        # Group 2: Passed through without modification
        self.group2 = tf.keras.layers.Lambda(lambda x: x)

        # Merge outputs from both groups
        self.concat = tf.keras.layers.Concatenate(axis=-1)

        # Block 2: Channel shuffling
        self.reshape = tf.keras.layers.Reshape(target_shape=(-1, 32, 4, 8))
        self.permute = tf.keras.layers.Permute(dims=(-1, 32, 4, 8))
        self.reshape_shuffled = tf.keras.layers.Reshape(target_shape=(-1, 32, 4, 8))

        # Final classification layer
        self.flatten = tf.keras.layers.Flatten()
        self.dense = tf.keras.layers.Dense(units=10, activation='softmax')

    def call(self, inputs):
        # Block 1: Initial convolution
        x = self.conv1(inputs)

        # Split into two groups
        group1, group2 = self.split(x)

        # Process group1
        group1 = self.conv1_1x1(group1)
        group1 = self.depthwise_conv(group1)
        group1 = self.conv1_1x1_after(group1)

        # Process group2
        group2 = self.group2(group2)

        # Merge outputs
        merged = self.concat([group1, group2])

        # Block 2: Channel shuffling
        reshaped = self.reshape(merged)
        permuted = self.permute(reshaped)
        reshaped_shuffled = self.reshape_shuffled(permuted)

        # Classification
        flattened = self.flatten(reshaped_shuffled)
        output = self.dense(flattened)

        return output

if __name__ == "__main__":
    # Create a dummy input tensor with shape (batch_size, height, width, channels)
    # MNIST images are 28x28 with 1 channel
    input_tensor = tf.random.normal(shape=(1, 28, 28, 1))
    model = DL_Model()
    output_tensor = model(input_tensor)
    print("Input shape:", input_tensor.shape)
    print("Output shape:", output_tensor.shape)