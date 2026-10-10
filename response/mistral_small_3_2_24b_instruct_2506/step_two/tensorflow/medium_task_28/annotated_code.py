import tensorflow as tf
from tensorflow.keras.layers import LayerNormalization, Dense, Flatten
from tensorflow.keras.models import Model

class DL_Model(Model):
    def __init__(self):
        super(DL_Model, self).__init__()
        # Attention weights generation
        self.attention_conv = tf.keras.layers.Conv2D(filters='mask', kernel_size=(1, 1), activation=None)
        self.softmax = tf.keras.layers.Activation('softmax')

        # Dimensionality reduction
        self.reduce_conv = tf.keras.layers.Conv2D(filters=32//3, kernel_size=(1, 1), activation=None)
        self.layer_norm = LayerNormalization(axis=-1, epsilon=1e-6)
        self.relu = tf.keras.layers.Activation('relu')

        # Dimensionality restoration
        self.restore_conv = tf.keras.layers.Conv2D(filters=32, kernel_size=(1, 1), activation=None)

        # Classification head
        self.flatten = Flatten()
        self.dense = Dense(units=10, activation='softmax')

    def call(self, inputs):
        # the initial shape of inputs is: (1, 32, 32, 3)
        # Generate attention weights
        attention_weights = self.attention_conv(inputs)  # output tensor shape of this layer should be (1, 32, 32, 3)
        attention_weights = self.softmax(attention_weights)

        # Apply attention weights
        contextual_features = tf.math.multiply(inputs, attention_weights)

        # Reduce dimensionality
        reduced_features = self.reduce_conv(contextual_features)
        normalized_features = self.layer_norm(reduced_features)
        activated_features = self.relu(normalized_features)

        # Restore dimensionality
        restored_features = self.restore_conv(activated_features)

        # Residual connection
        output = tf.math.add(inputs, restored_features)

        # Classification
        flattened = self.flatten(output)
        logits = self.dense(flattened)

        return logits

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal(shape=(1, 32, 32, 3))  # CIFAR-10 image shape
    output_tensor = model(input_tensor)
    print("Output shape:", output_tensor.shape)