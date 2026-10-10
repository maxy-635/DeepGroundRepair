import tensorflow as tf
from tensorflow.keras.layers import SeparableConv2D, LayerNormalization, Dense, Add, Flatten
from tensorflow.keras.models import Model

class DL_Model(Model):
    def __init__(self):
        super(DL_Model, self).__init__()
        # Spatial feature extraction with depthwise separable convolution
        self.separable_conv = SeparableConv2D(
            filters=64,
            kernel_size=(7, 7),
            strides=(1, 1),
            padding='same',
            activation='relu'
        )
        # Layer normalization for training stability
        self.layer_norm = LayerNormalization(
            axis=-1,
            epsilon=1e-6,
            center=True,
            scale=True
        )
        # Fully connected layers for channel-wise feature transformation
        self.fc1 = Dense(
            units=64,
            activation='relu'
        )
        self.fc2 = Dense(
            units=64,
            activation='relu'
        )
        # Final fully connected layers for classification
        self.fc3 = Dense(
            units=128,
            activation='relu'
        )
        self.fc4 = Dense(
            units=10,
            activation='softmax'
        )
        # Addition operation
        self.add = Add()

    def call(self, inputs):
        # Extract spatial features
        x = self.separable_conv(inputs)
        x = self.layer_norm(x)

        # Channel-wise feature transformation
        x_flat = Flatten()(x)
        x_fc = self.fc1(x_flat)
        x_fc = self.fc2(x_fc)
        x_fc = tf.reshape(x_fc, (-1, 32, 32, 64))  # Reshape to match input dimensions

        # Combine original input with processed features
        combined = self.add([inputs, x_fc])

        # Final classification
        x_flat = Flatten()(combined)
        x = self.fc3(x_flat)
        output = self.fc4(x)

        return output

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal(shape=(1, 32, 32, 3))  # CIFAR-10 input shape
    output_tensor = model(input_tensor)
    print("Model output shape:", output_tensor.shape)