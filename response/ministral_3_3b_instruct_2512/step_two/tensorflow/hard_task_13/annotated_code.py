import tensorflow as tf

class DL_Model(tf.keras.Model):
    def __init__(self):
        super(DL_Model, self).__init__()
        # First block: four parallel branches
        self.conv1x1 = tf.keras.layers.Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')
        self.conv3x3 = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')
        self.conv5x5 = tf.keras.layers.Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')
        self.max_pool = tf.keras.layers.MaxPool2D(pool_size='mask', strides='mask', padding='mask')

        # Second block: Global Average Pooling and fully connected layers
        self.global_avg_pool = tf.keras.layers.GlobalAveragePooling2D(keepdims=False)
        self.fc1 = tf.keras.layers.Dense(units=64, activation='relu')
        self.fc2 = tf.keras.layers.Dense(units=10)  # Assuming 10 output classes for CIFAR-10

        # Weight reshaping and multiplication
        self.reshape = tf.keras.layers.Reshape(target_shape=(-1, 64, 1, 1))
        self.multiply = tf.keras.layers.Multiply()

    def call(self, inputs):
        # the initial shape of inputs is: (1, 32, 32, 3)
        # First block: parallel branches
        conv1x1_out = self.conv1x1(inputs)  # output tensor shape: (1, 32, 32, 64)
        conv3x3_out = self.conv3x3(inputs)  # output tensor shape: (1, 32, 32, 64)
        conv5x5_out = self.conv5x5(inputs)  # output tensor shape: (1, 32, 32, 64)
        max_pool_out = self.max_pool(inputs)  # output tensor shape of this layer should be (1, 32, 32, 3)

        # Concatenate outputs
        concatenated = tf.keras.layers.concatenate([conv1x1_out, conv3x3_out, conv5x5_out, max_pool_out], axis=3)

        # Second block: Global Average Pooling and fully connected layers
        pooled_features = self.global_avg_pool(concatenated)
        pooled_features = tf.reshape(pooled_features, (-1, 64))

        # Reshape weights and multiply
        weights = self.fc1(pooled_features)
        weights = self.reshape(weights)
        weighted_features = self.multiply([concatenated, weights])

        # Final fully connected layer
        output = self.fc2(weighted_features)
        return output

if __name__ == "__main__":
    # Create a dummy input tensor with shape (batch_size, height, width, channels)
    # Assuming CIFAR-10 images are 32x32 with 3 channels
    input_tensor = tf.random.normal(shape=(1, 32, 32, 3))
    model = DL_Model()
    output_tensor = model(input_tensor)
    print("Input shape:", input_tensor.shape)
    print("Output shape:", output_tensor.shape)