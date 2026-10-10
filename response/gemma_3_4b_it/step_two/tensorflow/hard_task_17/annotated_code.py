import tensorflow as tf

class DL_Model(tf.keras.Model):
    def __init__(self):
        super(DL_Model, self).__init__()
        self.block1 = tf.keras.Sequential([
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(10)
        ])
        self.block2 = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, (3, 3), strides=(1, 1), padding='same', activation='relu'),
            tf.keras.layers.Conv2D(64, (3, 3), strides=(1, 1), padding='same', activation='relu'),
            tf.keras.layers.MaxPool2D((2, 2), strides=(2, 2))
        ])
        self.dense = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(10, activation='softmax')
        ])

    def call(self, inputs):
        # the initial shape of inputs is: (1, 32, 32, 3)
        # Block 1
        block1_output = self.block1(inputs)  # output tensor shape: (1, 10)
        reshaped_block1 = tf.reshape(block1_output, (-1, 10))  # output tensor shape: (1, 10)
        weighted_feature = tf.linalg.matmul(reshaped_block1, inputs)

        # Block 2
        block2_output = self.block2(inputs)

        # Fusion
        combined_output = tf.add(weighted_feature, block2_output)

        # Classification
        output_tensor = self.dense(combined_output)

        return output_tensor


if __name__ == "__main__":
    # Dummy input tensor for validation
    input_tensor = tf.random.normal((1, 32, 32, 3))  # Example input shape
    model = DL_Model()
    output_tensor = model(input_tensor)
    print("Output shape:", output_tensor.shape)