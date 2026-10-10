import tensorflow as tf

class DL_Model(tf.keras.Model):
    def __init__(self):
        super(DL_Model, self).__init__()
        # Initial Convolutional Layer
        self.conv1 = tf.keras.layers.Conv2D(
            filters=32,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='same',
            activation='relu'
        )

        # Path 1: Global Average Pooling + Dense Layers
        self.global_avg_pool1 = tf.keras.layers.GlobalAveragePooling2D(keepdims=True)
        self.dense1 = tf.keras.layers.Dense(
            units=16,
            activation='relu'
        )
        self.dense2 = tf.keras.layers.Dense(
            units=1,
            activation='sigmoid'
        )

        # Path 2: Global Max Pooling + Dense Layers
        self.global_max_pool1 = tf.keras.layers.GlobalMaxPool2D()
        self.dense3 = tf.keras.layers.Dense(
            units=16,
            activation='relu'
        )
        self.dense4 = tf.keras.layers.Dense(
            units=1,
            activation='sigmoid'
        )

    def call(self, inputs):
        # Initial Convolutional Layer
        x = self.conv1(inputs)

        # Path 1: Global Average Pooling
        x1 = self.global_avg_pool1(x)
        x1 = self.dense1(x1)
        x1 = self.dense2(x1)

        # Path 2: Global Max Pooling
        x2 = self.global_max_pool1(x)
        x2 = self.dense3(x2)
        x2 = self.dense4(x2)

        # Add the two paths
        x = tf.keras.layers.Add()( [x1, x2])

        # Activation function
        x = tf.keras.layers.Activation('relu')(x)

        # Channel Attention
        channel_attention = x
        
        # Apply channel attention
        x = tf.keras.layers.Multiply()( [x, channel_attention])
        

        # Average Pooling
        x = tf.keras.layers.AveragePooling2D(
            pool_size=(2, 2),
            strides=(1, 1),
            padding='same'
        )(x)

        # Max Pooling
        x = tf.keras.layers.MaxPool2D(
            pool_size=(2, 2),
            strides=(1, 1),
            padding='same'
        )(x)

        # Concatenate along the channel dimension
        x = tf.keras.layers.concatenate(
            [x, x]
        )([x1, x2])

        # Flatten
        x = tf.keras.layers.Flatten()(x)

        # Fully Connected Layer
        x = tf.keras.layers.Dense(
            units=10,
            activation='softmax'
        )(x)

        return x


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal((1, 32, 32, 3))  # Example input shape: (batch_size, height, width, channels)
    output_tensor = model(input_tensor)
    print("Output shape:", output_tensor.shape)