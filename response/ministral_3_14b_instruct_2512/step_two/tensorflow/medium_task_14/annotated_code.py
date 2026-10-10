import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, ReLU, Add, Dense, Flatten, concatenate
from tensorflow.keras.models import Model

class DL_Model(Model):
    def __init__(self):
        super(DL_Model, self).__init__()
        # Sequential blocks with Conv2D, BatchNormalization, and ReLU
        self.conv_block1 = tf.keras.Sequential([
            Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same'),
            BatchNormalization(axis=-1),
            ReLU()
        ])

        self.conv_block2 = tf.keras.Sequential([
            Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same'),
            BatchNormalization(axis=-1),
            ReLU()
        ])

        self.conv_block3 = tf.keras.Sequential([
            Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same'),
            BatchNormalization(axis=-1),
            ReLU()
        ])

        # Parallel branch
        self.parallel_conv = Conv2D(filters='mask', kernel_size=(1, 1), strides=(1, 1), padding='same')

        # Fully connected layers
        self.fc1 = Dense(units=512, activation='relu')
        self.fc2 = Dense(units=10, activation='softmax')  # CIFAR-10 has 10 classes

    def call(self, inputs):
        # the initial shape of inputs is: (1, 32, 32, 3)
        # Process through sequential blocks
        block1_out = self.conv_block1(inputs)  # output tensor shape: (1, 32, 32, 64)
        block2_out = self.conv_block2(block1_out)  # output tensor shape: (1, 32, 32, 128)
        block3_out = self.conv_block3(block2_out)  # output tensor shape: (1, 32, 32, 256)

        # Process through parallel branch
        parallel_out = self.parallel_conv(inputs)  # output tensor shape of this layer should be (1, 32, 32, 256)

        # Add all outputs together
        combined = Add()([block3_out, parallel_out])

        # Flatten and pass through fully connected layers
        flattened = Flatten()(combined)
        fc1_out = self.fc1(flattened)
        output = self.fc2(fc1_out)

        return output

if __name__ == "__main__":
    model = DL_Model()
    # CIFAR-10 images are 32x32 with 3 channels
    input_tensor = tf.random.normal(shape=(1, 32, 32, 3))
    output_tensor = model(input_tensor)
    print("Input shape:", input_tensor.shape)
    print("Output shape:", output_tensor.shape)