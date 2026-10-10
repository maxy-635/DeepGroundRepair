import tensorflow as tf

class DL_Model(tf.keras.Model):
    def __init__(self):
        super(DL_Model, self).__init__()
        # Main path layers
        self.conv1 = tf.keras.layers.Conv2D(32, kernel_size=(3, 3), strides=(1, 1), padding='same')
        self.conv2 = tf.keras.layers.Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding='same')
        self.conv3 = tf.keras.layers.Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding='same')

        # Branch path layer
        self.branch_conv = tf.keras.layers.Conv2D(32, kernel_size=(1, 1), strides=(1, 1), padding='same')

        # Concatenation and classification layers
        self.flatten = tf.keras.layers.Flatten()
        self.dense = tf.keras.layers.Dense(10, activation='softmax')

    def call(self, inputs):
        # Split the input into three groups along the last dimension
        # For CIFAR-10 (3 channels), we'll split into three groups of 1 channel each
        # This is done by splitting along the channel dimension (axis=3)
        split_input = tf.split(inputs, num_or_size_splits=3, axis=3)

        # Process each group
        group1 = split_input[0]  # unchanged
        group2 = self.conv1(split_input[1])  # feature extraction via 3x3 conv
        group3 = split_input[2]  # unchanged

        # Combine group2 and group3
        combined = tf.concat([group2, group3], axis=3)

        # Apply another 3x3 convolution
        main_output = self.conv2(combined)
        main_output = self.conv3(main_output)

        # Branch path
        branch_output = self.branch_conv(inputs)

        # Concatenate outputs from main and branch paths
        # First flatten the main output to match branch output dimensions
        main_flattened = self.flatten(main_output)
        branch_flattened = self.flatten(branch_output)

        # Concatenate along the last dimension
        concatenated = tf.concat([main_flattened, branch_flattened], axis=1)

        # Flatten and pass through dense layer
        output = self.flatten(concatenated)
        output = self.dense(output)

        return output

if __name__ == "__main__":
    # Create a dummy input tensor with shape (batch_size, height, width, channels)
    # For CIFAR-10, we'll use (1, 32, 32, 3)
    input_tensor = tf.random.normal((1, 32, 32, 3))
    model = DL_Model()
    output_tensor = model(input_tensor)
    print("Input shape:", input_tensor.shape)
    print("Output shape:", output_tensor.shape)