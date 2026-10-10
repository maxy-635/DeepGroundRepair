import tensorflow as tf

class DL_Model(tf.keras.Model):
    def __init__(self):
        super(DL_Model, self).__init__()
        # Initial convolution layer
        self.conv1 = tf.keras.layers.Conv2D(
            filters=16,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='same',
        )
        # Batch Normalization
        self.bn1 = tf.keras.layers.BatchNormalization(axis=-1)
        # ReLU activation
        self.relu1 = tf.keras.layers.ReLU()

        # Residual Block 1
        self.residual_block1 = tf.keras.Sequential([
            tf.keras.layers.Conv2D(
                filters=16,
                kernel_size=(3, 3),
                strides=(1, 1),
                padding='same',
            ),
            tf.keras.layers.BatchNormalization(axis=-1),
            tf.keras.layers.ReLU(),
        ])

        # Residual Block 2
        self.residual_block2 = tf.keras.Sequential([
            tf.keras.layers.Conv2D(
                filters=16,
                kernel_size=(3, 3),
                strides=(1, 1),
                padding='same',
            ),
            tf.keras.layers.BatchNormalization(axis=-1),
            tf.keras.layers.ReLU(),
        ])

        # Global Branch
        self.global_conv = tf.keras.layers.Conv2D(
            filters=16,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='same',
        )

        # Average Pooling
        self.avg_pool = tf.keras.layers.AveragePooling2D(
            pool_size=(2, 2),
            strides=(2, 2),
            padding='same',
        )

        # Dense layer
        self.dense = tf.keras.layers.Dense(10, activation='softmax')

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.relu1(x)

        # Residual Block 1
        residual1 = self.residual_block1(x)
        x = x + residual1

        # Residual Block 2
        residual2 = self.residual_block2(x)
        x = x + residual2

        # Global Branch
        global_branch = self.global_conv(x)

        x = x + global_branch

        x = self.avg_pool(x)
        x = tf.keras.layers.Flatten()(x)
        x = self.dense(x)

        return x

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal((1, 32, 32, 3))  # Example input shape
    output_tensor = model(input_tensor)
    print(output_tensor.shape)