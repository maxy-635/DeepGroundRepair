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
        self.restore_conv = Conv2D(filters=3, kernel_size=(1, 1), strides=(1, 1), padding='valid')

        # Classification layers
        self.flatten = Flatten()
        self.classifier = Dense(units=10, activation=None)

    def call(self, inputs):
        # the initial shape of inputs is: (1, 32, 32, 3)
        # Attention mechanism
        attention_weights = self.attention_conv(inputs)  # output tensor shape: (1, 32, 32, 1)
        attention_weights = self.softmax(attention_weights)  # output tensor shape: (1, 32, 32, 1)
        context_features = Multiply()([inputs, attention_weights])  # output tensor shape: (1, 32, 32, 3)

        # Dimensionality reduction and restoration
        reduced_features = self.reduce_conv(inputs)  # output tensor shape: (1, 32, 32, 32)
        reduced_features = self.norm(reduced_features)  # output tensor shape: (1, 32, 32, 32)
        reduced_features = self.reduce_relu(reduced_features)  # output tensor shape: (1, 32, 32, 32)
        restored_features = self.restore_conv(reduced_features)  # output tensor shape of this layer should be (1, 32, 32, 3)

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