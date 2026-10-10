import tensorflow as tf

class DL_Model(tf.keras.Model):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Block 1
        self.global_avg_pool = tf.keras.layers.GlobalAveragePooling2D(keepdims=True)
        self.dense1 = tf.keras.layers.Dense(units=32, activation='relu')
        self.dense2 = tf.keras.layers.Dense(units=32, activation='relu')
        self.reshape = tf.keras.layers.Reshape(target_shape=(1, 1, 32))
        

        # Block 2
        self.conv1 = tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=1, padding='same', activation='relu')
        self.conv2 = tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=1, padding='same', activation='relu')
        self.max_pool = tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding='same')

        # Output layers
        self.dense3 = tf.keras.layers.Dense(units=128, activation='relu')
        self.dense4 = tf.keras.layers.Dense(units=10, activation='softmax')

    def call(self, inputs):
        # the initial shape of inputs is: (1, 32, 32, 3)
        # Block 1
        x1 = self.global_avg_pool(inputs)  # output tensor shape: (1, 1, 1, 3)
        x1 = self.dense1(x1)  # output tensor shape: (1, 1, 1, 32)
        x1 = self.dense2(x1)  # output tensor shape: (1, 1, 1, 32)
        x1 = self.reshape(x1)
        weighted_feature = tf.keras.layers.Multiply()([inputs, x1])

        # Block 2
        x2 = self.conv1(inputs)
        x2 = self.conv2(x2)
        x2 = self.max_pool(x2)

        # Branch from Block 1 to Block 2
        x2 = tf.keras.layers.Add()([x2, weighted_feature])

        # Output layers
        x = x2
        x = tf.keras.layers.Flatten()(x)
        x = self.dense3(x)
        output = self.dense4(x)

        return output


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal(shape=(1, 32, 32, 3))
    output_tensor = model(input_tensor)
    print("Input shape:", input_tensor.shape)
    print("Output shape:", output_tensor.shape)