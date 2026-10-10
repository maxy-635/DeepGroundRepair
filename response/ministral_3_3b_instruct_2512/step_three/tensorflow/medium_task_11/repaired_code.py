import tensorflow as tf

class DL_Model(tf.keras.Model):
    def __init__(self):
        super(DL_Model, self).__init__()
        # Initial convolutional layer
        self.conv_layer = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same')

        # Channel attention path 1: Global Average Pooling + Dense layers
        self.avg_pool_path = tf.keras.Sequential([
            tf.keras.layers.GlobalAveragePooling2D(keepdims=True),
            tf.keras.layers.Dense(units=16, activation='relu'),
            tf.keras.layers.Dense(units=1)
        ])

        # Channel attention path 2: Global Max Pooling + Dense layers
        self.max_pool_path = tf.keras.Sequential([
            tf.keras.layers.GlobalMaxPool2D(keepdims=True),
            tf.keras.layers.Dense(units=16, activation='relu'),
            tf.keras.layers.Dense(units=1)
        ])

        # Spatial attention path - modified to handle 32-channel output
        self.spatial_pool = tf.keras.Sequential([
            tf.keras.layers.GlobalAveragePooling2D(keepdims=True),  # Changed to keepdims=True
            tf.keras.layers.GlobalMaxPool2D(keepdims=True)        # Changed to keepdims=True
        ])

        # Fully connected layer for final output
        self.fc_layer = tf.keras.layers.Dense(units=10)

    def call(self, inputs):
        # Initial convolution
        x = self.conv_layer(inputs)

        # Channel attention
        avg_pool_out = self.avg_pool_path(x)
        max_pool_out = self.max_pool_path(x)
        attention_weights = tf.keras.layers.Multiply()([avg_pool_out, max_pool_out])
        attention_weights = tf.keras.layers.Activation('sigmoid')(attention_weights)
        channel_attention = tf.keras.layers.Multiply()([x, attention_weights])

        # Spatial attention - modified to handle 32-channel output
        spatial_avg = self.spatial_pool(x)[0]
        spatial_max = self.spatial_pool(x)[1]
        spatial_features = tf.keras.layers.concatenate([spatial_avg, spatial_max], axis=-1)

        # Combine channel and spatial features
        combined_features = tf.keras.layers.Multiply()([channel_attention, spatial_features])

        # Flatten and final classification
        flattened = tf.keras.layers.Flatten()(combined_features)
        output = self.fc_layer(flattened)

        return output

if __name__ == "__main__":
    # Create model instance
    model = DL_Model()

    # Create dummy input tensor with shape (batch_size, height, width, channels)
    # Using CIFAR-10 dimensions: 32x32 RGB images
    input_tensor = tf.random.normal(shape=(1, 32, 32, 3))
    output_tensor = model(input_tensor)
    print("Input shape:", input_tensor.shape)
    print("Output shape:", output_tensor.shape)