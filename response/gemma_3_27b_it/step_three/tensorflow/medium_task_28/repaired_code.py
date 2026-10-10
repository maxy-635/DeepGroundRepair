import tensorflow as tf

class DL_Model(tf.keras.Model):
    def __init__(self):
        super(DL_Model, self).__init__()
        # Attention Mechanism
        self.attention_conv = tf.keras.layers.Conv2D(filters=1, kernel_size=1, strides=1, padding='same', activation=None)
        self.attention_softmax = tf.keras.layers.Softmax(axis=-1)
        # Dimensionality Reduction and Restoration
        self.reduction_conv = tf.keras.layers.Conv2D(filters=32, kernel_size=1, strides=1, padding='same', activation=None)  # Reduce to 1/3 of original (assuming 96 filters initially)
        self.layer_norm = tf.keras.layers.LayerNormalization(axis=-1, epsilon=1e-6)
        self.relu = tf.keras.layers.ReLU(max_value=None, negative_slope=0.0)
        self.restoration_conv = tf.keras.layers.Conv2D(filters=3, kernel_size=1, strides=1, padding='same', activation=None)
        # Classification Head
        self.flatten = tf.keras.layers.Flatten()
        self.dense = tf.keras.layers.Dense(units=10, activation='softmax')

    def call(self, inputs):
        # the initial shape of inputs is: (1, 32, 32, 3)
        # Attention Mechanism
        attention_weights = self.attention_conv(inputs)  # output tensor shape: (1, 32, 32, 1)
        attention_weights = self.attention_softmax(attention_weights)  # output tensor shape: (1, 32, 32, 1)
        contextual_features = tf.keras.layers.Multiply()([inputs, attention_weights])  # output tensor shape: (1, 32, 32, 3)

        # Dimensionality Reduction and Restoration
        reduced_features = self.reduction_conv(contextual_features)  # output tensor shape: (1, 32, 32, 32)
        normalized_features = self.layer_norm(reduced_features)  # output tensor shape: (1, 32, 32, 32)
        activated_features = self.relu(normalized_features)  # output tensor shape: (1, 32, 32, 32)
        restored_features = self.restoration_conv(activated_features)  # output tensor shape of this layer should be (1, 32, 32, 3)
        residual_connection = tf.keras.layers.Add()([inputs, restored_features])

        # Classification Head
        flattened_features = self.flatten(residual_connection)
        outputs = self.dense(flattened_features)
        return outputs


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal(shape=(1, 32, 32, 3))  # Example input tensor (batch_size=1, height=32, width=32, channels=3)
    output_tensor = model(input_tensor)
    print(output_tensor.shape)