import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, ReLU, Add, Dense, Flatten

class DL_Model(tf.keras.Model):
    def __init__(self):
        super(DL_Model, self).__init__()
        # Sequential blocks
        self.block1 = tf.keras.Sequential([
            Conv2D(filters='mask', kernel_size=(3, 3), strides=(1, 1), padding='same'),
            BatchNormalization(axis=-1),
            ReLU()
        ])

        self.block2 = tf.keras.Sequential([
            Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding='same'),
            BatchNormalization(axis=-1),
            ReLU()
        ])

        self.block3 = tf.keras.Sequential([
            Conv2D(128, kernel_size=(3, 3), strides=(1, 1), padding='same'),
            BatchNormalization(axis=-1),
            ReLU()
        ])

        # Parallel branch
        self.parallel_branch = tf.keras.Sequential([
            Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding='same'),
            BatchNormalization(axis=-1),
            ReLU()
        ])

        # Fully connected layers
        self.fc1 = Dense(512, activation='relu')
        self.fc2 = Dense(10, activation='softmax')

    def call(self, inputs):
        # the initial shape of inputs is: (1, 32, 32, 3)
        # Sequential blocks outputs
        block1_out = self.block1(inputs)  # output tensor shape of this layer should be (1, 32, 32, 64)
        block2_out = self.block2(block1_out)  # output tensor shape: (1, 32, 32, 64)
        block3_out = self.block3(block2_out)  # output tensor shape: (1, 32, 32, 128)

        # Parallel branch output
        parallel_out = self.parallel_branch(inputs)  # output tensor shape: (1, 32, 32, 64)

        # Add all outputs
        combined = Add()([block3_out, block2_out, block1_out, parallel_out])

        # Flatten and pass through FC layers
        flattened = Flatten()(combined)
        output = self.fc1(flattened)
        output = self.fc2(output)

        return output

if __name__ == "__main__":
    # Create model instance
    model = DL_Model()

    # Create dummy input tensor with shape (batch_size, height, width, channels)
    # For CIFAR-10, height=32, width=32, channels=3
    input_tensor = tf.random.normal((1, 32, 32, 3))

    # Validate the model
    output_tensor = model(input_tensor)
    print("Output shape:", output_tensor.shape)