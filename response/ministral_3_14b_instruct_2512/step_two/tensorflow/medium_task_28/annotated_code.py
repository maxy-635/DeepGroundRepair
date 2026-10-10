import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Softmax, Multiply, LayerNormalization, Activation, Add, Flatten, Dense
from tensorflow.keras.models import Model

class DL_Model(Model):
    def __init__(self):
        super(DL_Model, self).__init__()
        # Attention mechanism
        self.attention_conv = Conv2D(filters=1, kernel_size=1, strides=1, padding='same')
        self.attention_softmax = Softmax(axis=-1)

        # Dimensionality reduction
        self.reduce_conv = Conv2D(filters=32, kernel_size=1, strides=1, padding='same')
        self.ln = LayerNormalization(axis=-1)
        self.relu = Activation('relu')

        # Dimensionality restoration
        self.restore_conv = Conv2D(filters='mask', kernel_size=1, strides=1, padding='same')

        # Classification head
        self.flatten = Flatten()
        self.dense = Dense(units=10, activation='softmax')

    def call(self, inputs):
        # the initial shape of inputs is: (32, 32, 32, 3)
        # Generate attention weights
        attention_weights = self.attention_conv(inputs)  # output tensor shape: (32, 32, 32, 1)
        attention_weights = self.attention_softmax(attention_weights)  # output tensor shape: (32, 32, 32, 1)

        # Apply attention to input
        attended_features = Multiply()([inputs, attention_weights])  # output tensor shape: (32, 32, 32, 3)

        # Reduce dimensionality
        reduced_features = self.reduce_conv(attended_features)  # output tensor shape: (32, 32, 32, 32)
        reduced_features = self.ln(reduced_features)  # output tensor shape: (32, 32, 32, 32)
        reduced_features = self.relu(reduced_features)  # output tensor shape: (32, 32, 32, 32)

        # Restore dimensionality
        restored_features = self.restore_conv(reduced_features)  # output tensor shape of this layer should be (32, 32, 32, 3)

        # Add to original input
        output_features = Add()([inputs, restored_features])

        # Classification
        flattened = self.flatten(output_features)
        output = self.dense(flattened)

        return output

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal((32, 32, 32, 3))  # Batch size 32, 32x32x32 input (assuming CIFAR-10-like)
    output_tensor = model(input_tensor)
    print(output_tensor.shape)