import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, ReLU, Concatenate, Flatten, Dense

class DL_Model(tf.keras.Model):
    def __init__(self):
        super(DL_Model, self).__init__()
        # First convolutional block
        self.conv1 = tf.keras.Sequential([
            Conv2D(32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation=None),
            BatchNormalization(axis=-1),
            ReLU(max_value=None, negative_slope=0.0)
        ])

        # Second convolutional block
        self.conv2 = tf.keras.Sequential([
            Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation=None),
            BatchNormalization(axis=-1),
            ReLU(max_value=None, negative_slope=0.0)
        ])

        # Third convolutional block
        self.conv3 = tf.keras.Sequential([
            Conv2D(128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation=None),
            BatchNormalization(axis=-1),
            ReLU(max_value=None, negative_slope=0.0)
        ])

        # Concatenate layers
        self.concatenate = Concatenate(axis=-1)

    def call(self, inputs):
        # First block output
        x = self.conv1(inputs)

        # Second block output
        y = self.conv2(inputs)

        # Third block output
        z = self.conv3(inputs)

        # Concatenate outputs along channel dimension
        output = self.concatenate([x, y, z])

        # Flatten and fully connected layers
        output = Flatten()(output)
        output = Dense(128, activation=None)(output)
        output = Dense(10, activation='softmax')(output)

        return output

if __name__ == "__main__":
    # Create model instance
    model = DL_Model()

    # Create dummy input tensor with shape (batch_size, height, width, channels)
    # For CIFAR-10, height=32, width=32, channels=3
    input_tensor = tf.random.normal((1, 32, 32, 3))

    # Forward pass
    output_tensor = model(input_tensor)
    print("Output shape:", output_tensor.shape)