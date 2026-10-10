import tensorflow as tf

class DL_Model(tf.keras.Model):
    def __init__(self):
        super(DL_Model, self).__init__()
        # Main path with three branches
        self.conv1x1 = tf.keras.layers.Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same')
        self.conv3x3 = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same')
        self.conv5x5 = tf.keras.layers.Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same')

        # Branch path
        self.branch_conv = tf.keras.layers.Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same')

        # Concatenation layer
        self.concatenate = tf.keras.layers.Concatenate(axis=-1)

        # Batch normalization and dropout
        self.bn1 = tf.keras.layers.BatchNormalization(axis=-1)
        self.dropout = tf.keras.layers.Dropout(rate=0.2)

        # Fully connected layers
        self.flatten = tf.keras.layers.GlobalAveragePooling2D(keepdims=False)
        self.fc1 = tf.keras.layers.Dense(units=128, activation='relu')
        self.fc2 = tf.keras.layers.Dense(units=10, activation='softmax')

    def call(self, inputs):
        # Main path branches
        conv1x1_out = self.conv1x1(inputs)
        conv3x3_out = self.conv3x3(inputs)
        conv5x5_out = self.conv5x5(inputs)

        # Branch path
        branch_out = self.branch_conv(inputs)

        # Concatenate outputs
        concatenated = self.concatenate([conv1x1_out, conv3x3_out, conv5x5_out, branch_out])

        # Batch normalization and dropout
        normalized = self.bn1(concatenated)
        dropped = self.dropout(normalized)

        # Fully connected layers
        pooled = self.flatten(dropped)
        fc1_out = self.fc1(pooled)
        output = self.fc2(fc1_out)

        return output

if __name__ == "__main__":
    # Create model instance
    model = DL_Model()

    # Create dummy input tensor with shape (batch_size, height, width, channels)
    # For CIFAR-10, height=32, width=32, channels=3
    input_tensor = tf.random.normal(shape=(1, 32, 32, 3))

    # Forward pass
    output_tensor = model(input_tensor)
    print("Model output shape:", output_tensor.shape)