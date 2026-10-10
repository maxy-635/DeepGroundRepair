import tensorflow as tf

class DL_Model(tf.keras.Model):
    def __init__(self):
        super(DL_Model, self).__init__()

        # First block: Feature Extraction
        self.conv1 = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')
        self.conv2 = tf.keras.layers.Conv2D(filters=3, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')
        self.avg_pool = tf.keras.layers.AveragePooling2D(pool_size=(1, 1), strides=(1, 1), padding='same')

        # Second block: Channel Weight Generation and Application
        self.global_avg_pool = tf.keras.layers.GlobalAveragePooling2D(keepdims=False)
        self.dense1 = tf.keras.layers.Dense(units=64, activation='relu')
        self.dense2 = tf.keras.layers.Dense(units=64, activation='sigmoid')
        self.reshape = tf.keras.layers.Reshape(target_shape=(1, 1, 64))
        self.multiply = tf.keras.layers.Multiply()
        self.flatten = tf.keras.layers.Flatten()
        self.classifier = tf.keras.layers.Dense(units=10, activation='softmax')

        self.add = tf.keras.layers.Add()

    def call(self, inputs):
        # the initial shape of inputs is: (1, 32, 32, 3)
        # First block
        x = self.conv1(inputs)  # output tensor shape: (1, 32, 32, 32)
        x = self.conv2(x)  # output tensor shape of this layer should be (1, 32, 32, 3)
        x = self.avg_pool(x)  # output tensor shape of this layer should be (1, 32, 32, 64)

        # Addition with input
        x = self.add([inputs, x])

        # Second block
        y = self.global_avg_pool(x)
        y = self.dense1(y)
        y = self.dense2(y)
        y = self.reshape(y)
        x = self.multiply([x, y])
        x = self.flatten(x)
        output = self.classifier(x)

        return output


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal(shape=(1, 32, 32, 3))  # Dummy input tensor
    output_tensor = model(input_tensor)
    print("Input shape:", input_tensor.shape)
    print("Output shape:", output_tensor.shape)