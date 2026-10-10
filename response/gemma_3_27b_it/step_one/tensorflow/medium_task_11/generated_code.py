import tensorflow as tf

class DL_Model(tf.keras.Model):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Initial convolutional layer
        self.conv1 = tf.keras.layers.Conv2D(
            filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'
        )

        # Channel attention module
        self.gap = tf.keras.layers.GlobalAveragePooling2D(keepdims=True)
        self.gmp = tf.keras.layers.GlobalMaxPool2D(keepdims=True)
        self.dense1 = tf.keras.layers.Dense(units=64, activation='relu')
        self.dense2 = tf.keras.layers.Dense(units=32, activation='sigmoid')

        # Spatial feature extraction
        self.avg_pool = tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')
        self.max_pool = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')
        self.concat = tf.keras.layers.Concatenate(axis=-1)

        # Final layers
        self.flatten = tf.keras.layers.Flatten()
        self.dense3 = tf.keras.layers.Dense(units=10, activation='softmax')

    def call(self, inputs):
        # Initial convolution
        x = self.conv1(inputs)

        # Channel attention
        gap = self.gap(x)
        gmp = self.gmp(x)
        gap_dense1 = self.dense1(gap)
        gmp_dense1 = self.dense1(gmp)
        gap_dense2 = self.dense2(gap_dense1)
        gmp_dense2 = self.dense2(gmp_dense1)
        channel_attention = tf.keras.layers.Add()([gap_dense2, gmp_dense2])
        channel_attention = tf.keras.layers.Multiply()([x, channel_attention])

        # Spatial feature extraction
        avg_pool_out = self.avg_pool(channel_attention)
        max_pool_out = self.max_pool(channel_attention)
        spatial_features = self.concat([avg_pool_out, max_pool_out])

        # Combine channel and spatial features
        combined_features = tf.keras.layers.Multiply()([channel_attention, spatial_features])

        # Final layers
        flattened = self.flatten(combined_features)
        output = self.dense3(flattened)

        return output


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal(shape=(1, 32, 32, 3))
    output_tensor = model(input_tensor)
    print(output_tensor.shape)