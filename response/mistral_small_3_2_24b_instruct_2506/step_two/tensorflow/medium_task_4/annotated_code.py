import tensorflow as tf
from tensorflow.keras.layers import Conv2D, AveragePooling2D, Flatten, Dense
from tensorflow.keras.models import Sequential

class DL_Model(Sequential):
    def __init__(self):
        super(DL_Model, self).__init__()
        # Path1: Two blocks of convolution followed by average pooling
        self.path1 = Sequential([
            Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same'),
            AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'),
            Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same'),
            AveragePooling2D(pool_size='mask', strides='mask', padding='mask')
        ])

        # Path2: Single convolutional layer
        self.path2 = Sequential([
            Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')
        ])

        # Flatten and combine the outputs
        self.flatten = Flatten()
        self.dense = Dense(units=10, activation='softmax')

    def call(self, inputs):
        # the initial shape of inputs is: (1, 32, 32, 3)
        # Process inputs through both pathways
        path1_output = self.path1(inputs)  # output tensor shape of this layer should be (1, 32, 32, 64)
        path2_output = self.path2(inputs)  # output tensor shape: (1, 32, 32, 64)

        # Combine the outputs by addition
        combined_output = tf.math.add(path1_output, path2_output)

        # Flatten and pass through the dense layer
        flattened = self.flatten(combined_output)
        output = self.dense(flattened)

        return output

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal(shape=(1, 32, 32, 3))  # Dummy input tensor for CIFAR-10 (batch_size=1, height=32, width=32, channels=3)
    output_tensor = model(input_tensor)
    print("Output shape:", output_tensor.shape)  # Expected output shape: (1, 10)