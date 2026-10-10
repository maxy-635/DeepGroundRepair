import tensorflow as tf
from tensorflow.keras.layers import SeparableConv2D, Conv2D, Dropout, Flatten, Dense, Concatenate, GlobalAveragePooling2D

class DL_Model(tf.keras.Model):
    def __init__(self):
        super(DL_Model, self).__init__()
        # Branch 1: Depthwise separable convolution + 1x1 convolution + dropout
        self.branch1 = tf.keras.Sequential([
            SeparableConv2D(32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'),
            Conv2D(32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu'),
            Dropout(rate=0.2)
        ])

        # Branch 2: Depthwise separable convolution + 1x1 convolution + dropout
        self.branch2 = tf.keras.Sequential([
            SeparableConv2D(64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'),
            Conv2D(64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu'),
            Dropout(rate=0.2)
        ])

        # Branch 3: Depthwise separable convolution + 1x1 convolution + dropout
        self.branch3 = tf.keras.Sequential([
            SeparableConv2D(128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'),
            Conv2D(128, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu'),
            Dropout(rate=0.2)
        ])

        # Concatenate branches and process with fully connected layers
        self.concatenate = Concatenate(axis=-1)
        self.dense1 = Dense(256, activation='relu')
        self.dense2 = Dense(10, activation='softmax')

    def call(self, inputs):
        # Process each branch
        branch1_output = self.branch1(inputs)
        branch2_output = self.branch2(inputs)
        branch3_output = self.branch3(inputs)

        # Concatenate outputs
        concatenated = self.concatenate([branch1_output, branch2_output, branch3_output])

        # Process concatenated output
        output = self.dense1(concatenated)
        output = self.dense2(output)

        return output

if __name__ == "__main__":
    # Create model instance
    model = DL_Model()

    # Create dummy input tensor with shape (batch_size, height, width, channels)
    # MNIST images are 28x28 with 1 channel (grayscale)
    input_tensor = tf.random.normal((1, 28, 28, 1))

    # Forward pass to validate the model
    output_tensor = model(input_tensor)
    print("Model output shape:", output_tensor.shape)