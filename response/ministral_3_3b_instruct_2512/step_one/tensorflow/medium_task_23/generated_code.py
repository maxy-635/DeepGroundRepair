import tensorflow as tf
from tensorflow.keras.layers import Conv1D, Conv2D, MaxPool2D, AveragePooling2D, Flatten, Dense, Concatenate, Reshape, BatchNormalization

class DL_Model(tf.keras.Model):
    def __init__(self):
        super(DL_Model, self).__init__()
        # Path 1: Single 1x1 convolution
        self.path1 = tf.keras.Sequential([
            Conv2D(64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')
        ])

        # Path 2: 1x1 convolution followed by 1x7 and 7x1 convolutions
        self.path2 = tf.keras.Sequential([
            Conv2D(64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu'),
            Conv2D(64, kernel_size=(1, 7), strides=(1, 1), padding='same', activation='relu'),
            Conv2D(64, kernel_size=(7, 1), strides=(1, 1), padding='same', activation='relu')
        ])

        # Path 3: 1x1 convolution followed by two sets of 1x7 and 7x1 convolutions
        self.path3 = tf.keras.Sequential([
            Conv2D(64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu'),
            Conv2D(64, kernel_size=(1, 7), strides=(1, 1), padding='same', activation='relu'),
            Conv2D(64, kernel_size=(7, 1), strides=(1, 1), padding='same', activation='relu'),
            Conv2D(64, kernel_size=(1, 7), strides=(1, 1), padding='same', activation='relu'),
            Conv2D(64, kernel_size=(7, 1), strides=(1, 1), padding='same', activation='relu')
        ])

        # Path 4: Average pooling followed by 1x1 convolution
        self.path4 = tf.keras.Sequential([
            AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'),
            Conv2D(64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')
        ])

        # Concatenate all paths
        self.concatenate = Concatenate(axis=-1)

        # Flatten and dense layer
        self.flatten = Flatten()
        self.dense = Dense(10, activation='softmax')

    def call(self, inputs):
        path1_output = self.path1(inputs)
        path2_output = self.path2(inputs)
        path3_output = self.path3(inputs)
        path4_output = self.path4(inputs)

        concatenated = self.concatenate([path1_output, path2_output, path3_output, path4_output])
        flattened = self.flatten(concatenated)
        output = self.dense(flattened)

        return output

if __name__ == "__main__":
    # Create a dummy input tensor with shape (batch_size, height, width, channels)
    # For CIFAR-10, each sample has shape (32, 32, 3)
    input_tensor = tf.random.normal((1, 32, 32, 3))
    model = DL_Model()
    output_tensor = model(input_tensor)
    print("Input shape:", input_tensor.shape)
    print("Output shape:", output_tensor.shape)