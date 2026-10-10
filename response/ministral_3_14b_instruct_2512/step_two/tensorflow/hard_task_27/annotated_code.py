import tensorflow as tf
from tensorflow.keras.layers import Conv2D, DepthwiseConv2D, SeparableConv2D, LayerNormalization, Dense, Add, Flatten
from tensorflow.keras.models import Model

class DL_Model(Model):
    def __init__(self):
        super(DL_Model, self).__init__()
        self.separable_conv = SeparableConv2D(
            filters=64,
            kernel_size=(7, 7),
            strides=(1, 1),
            padding='same',
            activation='relu'
        )
        self.layer_norm = LayerNormalization(axis=-1)
        self.dense1 = Dense(units=64, activation='relu')
        self.dense2 = Dense(units=64, activation='relu')
        self.add = Add()
        self.flatten = Flatten()
        self.fc1 = Dense(units=64, activation='relu')
        self.fc2 = Dense(units=10, activation='softmax')

    def call(self, inputs):
        # the initial shape of inputs is: (32, 32, 32, 3)
        # Extract spatial features with depthwise separable convolution
        x = self.separable_conv(inputs)  # output tensor shape: (32, 32, 32, 64)
        x = self.layer_norm(x)  # output tensor shape: (32, 32, 32, 64)

        # Channel-wise feature transformation
        x = tf.reduce_mean(x, axis=(1, 2))  # output tensor shape: (32, 64)
        x = self.dense1(x)  # output tensor shape: (32, 64)
        x = self.dense2(x)  # output tensor shape: (32, 64)

        # Combine original input with processed features
        original_shape = inputs.shape  # output tensor shape: (unknown)
        inputs_reshaped = tf.reshape(inputs, (original_shape[0], -1))  # output tensor shape: (32, 3072)
        combined = self.add([inputs_skip, x])

        # Final classification
        x = self.fc1(combined)
        x = self.fc2(x)

        return x

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal((32, 32, 32, 3))  # Example input shape for CIFAR-10 (batch_size, height, width, channels)
    output_tensor = model(input_tensor)
    print(output_tensor.shape)