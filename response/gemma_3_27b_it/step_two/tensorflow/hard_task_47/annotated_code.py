import tensorflow as tf

class DL_Model(tf.keras.Model):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Block 1: Depthwise Separable Convolutions
        self.depthwise_1x1 = tf.keras.layers.DepthwiseConv2D(
            kernel_size=(1, 1),
            strides=(1, 1),
            padding='same',
            depth_multiplier=1,
            activation='relu'
        )
        self.bn_1x1 = tf.keras.layers.BatchNormalization(axis=-1)

        self.depthwise_3x3 = tf.keras.layers.DepthwiseConv2D(
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='same',
            depth_multiplier=1,
            activation='relu'
        )
        self.bn_3x3 = tf.keras.layers.BatchNormalization(axis=-1)

        self.depthwise_5x5 = tf.keras.layers.DepthwiseConv2D(
            kernel_size=(5, 5),
            strides=(1, 1),
            padding='same',
            depth_multiplier=1,
            activation='relu'
        )
        self.bn_5x5 = tf.keras.layers.BatchNormalization(axis=-1)
        
        # Block 2: Feature Extraction Branches
        self.branch_1 = tf.keras.Sequential([
            tf.keras.layers.Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu'),
            tf.keras.layers.BatchNormalization(axis=-1)
        ])
        
        self.branch_2 = tf.keras.Sequential([
            tf.keras.layers.Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu'),
            tf.keras.layers.BatchNormalization(axis=-1),
            tf.keras.layers.Conv2D(filters=32, kernel_size=(1, 7), strides=(1, 1), padding='same', activation='relu'),
            tf.keras.layers.BatchNormalization(axis=-1),
            tf.keras.layers.Conv2D(filters=32, kernel_size=(7, 1), strides=(1, 1), padding='same', activation='relu'),
            tf.keras.layers.BatchNormalization(axis=-1),
            tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'),
            tf.keras.layers.BatchNormalization(axis=-1)
        ])
        
        self.branch_3 = tf.keras.layers.AveragePooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')
        
        # Classification Layers
        self.flatten = tf.keras.layers.Flatten()
        self.dense_1 = tf.keras.layers.Dense(units=128, activation='relu')
        self.dense_2 = tf.keras.layers.Dense(units=10, activation='softmax')

    def call(self, inputs):
        # Block 1
        x1 = self.depthwise_1x1(inputs)
        x1 = self.bn_1x1(x1)
        x3 = self.depthwise_3x3(inputs)
        x3 = self.bn_3x3(x3)
        x5 = self.depthwise_5x5(inputs)
        x5 = self.bn_5x5(x5)
        x = tf.keras.layers.Concatenate(axis=-1)([x1, x3, x5])

        # Block 2
        b1 = self.branch_1(x)
        b2 = self.branch_2(x)
        b3 = self.branch_3(x)
        x = tf.keras.layers.Concatenate(axis=-1)([b1, b2, b3])

        # Classification
        x = self.flatten(x)
        x = self.dense_1(x)
        return self.dense_2(x)


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal(shape=(1, 32, 32, 3))  # CIFAR-10 input shape
    output_tensor = model(input_tensor)
    print(output_tensor.shape)