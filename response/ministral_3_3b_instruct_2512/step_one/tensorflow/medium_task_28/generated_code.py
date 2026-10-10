import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Softmax, Add, Flatten, Dense, LayerNormalization, ReLU, Multiply, Reshape

class DL_Model(tf.keras.Model):
    def __init__(self):
        super(DL_Model, self).__init__()
        # Attention mechanism with 1x1 convolution and softmax
        self.attention_conv = Conv2D(filters=1, kernel_size=(1, 1), strides=(1, 1), padding='valid')
        self.softmax = Softmax(axis=-1)

        # Dimensionality reduction and restoration
        self.reduce_conv = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='valid')
        self.norm = LayerNormalization(axis=-1)
        self.reduce_relu = ReLU()
        self.restore_conv = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='valid')

        # Classification layers
        self.flatten = Flatten()
        self.classifier = Dense(units=10, activation=None)

    def call(self, inputs):
        # Attention mechanism
        attention_weights = self.attention_conv(inputs)
        attention_weights = self.softmax(attention_weights)
        context_features = Multiply()([inputs, attention_weights])

        # Dimensionality reduction and restoration
        reduced_features = self.reduce_conv(inputs)
        reduced_features = self.norm(reduced_features)
        reduced_features = self.reduce_relu(reduced_features)
        restored_features = self.restore_conv(reduced_features)

        # Add original and processed features
        processed_features = Add()([inputs, restored_features])

        # Classification
        flattened = self.flatten(processed_features)
        output = self.classifier(flattened)

        return output

if __name__ == "__main__":
    # Create model instance
    model = DL_Model()

    # Create dummy input tensor for CIFAR-10 (32, 32, 3)
    input_tensor = tf.random.normal(shape=(1, 32, 32, 3), dtype=tf.float32)

    # Forward pass
    output_tensor = model(input_tensor)
    print("Input shape:", input_tensor.shape)
    print("Output shape:", output_tensor.shape)