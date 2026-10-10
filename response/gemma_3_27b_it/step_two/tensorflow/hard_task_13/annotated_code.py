import tensorflow as tf

class DL_Model(tf.keras.Model):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Block 1: Parallel branches
        self.conv1x1 = tf.keras.layers.Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')
        self.conv3x3 = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')
        self.conv5x5 = tf.keras.layers.Conv2D(filters='mask', kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')
        self.maxpool = tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=(1, 1), padding='same')

        # Block 2: Dimensionality reduction and feature weighting
        self.global_avg_pool = tf.keras.layers.GlobalAveragePooling2D(keepdims=False)
        self.dense1 = tf.keras.layers.Dense(units=32*32*3, activation='relu') # Assuming input image size is 32x32 with 3 channels
        self.dense2 = tf.keras.layers.Dense(units=32*32*3, activation=None)
        self.reshape = tf.keras.layers.Reshape(target_shape=(32, 32, 3))
        self.multiply = tf.keras.layers.Multiply()
        self.flatten = tf.keras.layers.Flatten()
        self.dense3 = tf.keras.layers.Dense(units=10, activation='softmax')


    def call(self, inputs):
        # the initial shape of inputs is: (1, 32, 32, 3)
        # Block 1
        branch1 = self.conv1x1(inputs)  # output tensor shape: (1, 32, 32, 32)
        branch2 = self.conv3x3(inputs)  # output tensor shape: (1, 32, 32, 32)
        branch3 = self.conv5x5(inputs)  # output tensor shape of this layer should be (1, 32, 32, 3)
        branch4 = self.maxpool(inputs)

        merged = tf.keras.layers.Concatenate(axis=-1)([branch1, branch2, branch3, branch4])

        # Block 2
        pooled = self.global_avg_pool(merged)  # output tensor shape: (1, 99)
        weights = self.dense1(pooled)  # output tensor shape: (1, 3072)
        weights = self.dense2(weights)  # output tensor shape: (1, 3072)
        reshaped_weights = self.reshape(weights)  # output tensor shape: (1, 32, 32, 3)
        weighted_features = self.multiply([merged, reshaped_weights])
        
        flattened = self.flatten(weighted_features)
        output = self.dense3(flattened)

        return output


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal(shape=(1, 32, 32, 3))  # Batch size of 1, 32x32 image, 3 channels
    output_tensor = model(input_tensor)
    print(output_tensor.shape)