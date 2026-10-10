import tensorflow as tf

class DL_Model(tf.keras.Model):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Initial Convolutional Layer
        self.conv1 = tf.keras.layers.Conv2D(
            filters=16,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="same",
            activation=None  # No activation here, handled in basic block
        )
        self.bn1 = tf.keras.layers.BatchNormalization(axis=-1)
        self.relu1 = tf.keras.layers.ReLU()

        # Basic Block 1
        self.conv2 = tf.keras.layers.Conv2D(
            filters=16,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="same",
            activation=None
        )
        self.bn2 = tf.keras.layers.BatchNormalization(axis=-1)
        self.relu2 = tf.keras.layers.ReLU()

        self.conv3 = tf.keras.layers.Conv2D(
            filters=16,
            kernel_size=(1, 1),
            strides=(1, 1),
            padding="same",
            activation=None
        )
        self.bn3 = tf.keras.layers.BatchNormalization(axis=-1)
        self.relu3 = tf.keras.layers.ReLU()

        # Basic Block 2
        self.conv4 = tf.keras.layers.Conv2D(
            filters=16,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="same",
            activation=None
        )
        self.bn4 = tf.keras.layers.BatchNormalization(axis=-1)
        self.relu4 = tf.keras.layers.ReLU()

        self.conv5 = tf.keras.layers.Conv2D(
            filters=16,
            kernel_size=(1, 1),
            strides=(1, 1),
            padding="same",
            activation=None
        )
        self.bn5 = tf.keras.layers.BatchNormalization(axis=-1)
        self.relu5 = tf.keras.layers.ReLU()


        # Average Pooling and Fully Connected Layer
        self.avg_pool = tf.keras.layers.AveragePooling2D(
            pool_size=(8, 8),
            strides=(8, 8),
            padding="valid"
        )
        self.flatten = tf.keras.layers.Flatten()
        self.dense = tf.keras.layers.Dense(
            units=10,
            activation="softmax"
        )

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.relu1(x)

        # Basic Block 1
        residual1 = self.conv2(x)
        residual1 = self.bn2(residual1)
        residual1 = self.relu2(residual1)

        residual1 = self.conv3(residual1)
        residual1 = self.bn3(residual1)

        x = tf.keras.layers.Add()([x, residual1])
        x = self.relu3(x)
        
        # Basic Block 2
        residual2 = self.conv4(x)
        residual2 = self.bn4(residual2)
        residual2 = self.relu4(residual2)

        residual2 = self.conv5(residual2)
        residual2 = self.bn5(residual2)

        x = tf.keras.layers.Add()([x, residual2])
        x = self.relu5(x)

        x = self.avg_pool(x)
        x = self.flatten(x)
        x = self.dense(x)
        return x


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal(shape=(1, 32, 32, 3))  # CIFAR-10 input shape
    output_tensor = model(input_tensor)
    print(output_tensor.shape)