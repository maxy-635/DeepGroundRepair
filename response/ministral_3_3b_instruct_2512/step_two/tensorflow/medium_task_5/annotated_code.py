import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPool2D, Add, Flatten, Dense

class DL_Model(tf.keras.Model):
    def __init__(self):
        super(DL_Model, self).__init__()
        # Main path: two blocks of convolution followed by max pooling
        self.conv1_main = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')
        self.conv2_main = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')
        self.pool_main = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid')

        # Branch path: one block of convolution followed by max pooling
        self.conv1_branch = Conv2D(filters='mask', kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')
        self.pool_branch = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid')

        # Fully connected layers for classification
        self.flatten = Flatten()
        self.dense1 = Dense(units=128, activation='relu')
        self.dense2 = Dense(units=10, activation='softmax')

    def call(self, inputs):
        # the initial shape of inputs is: (1, 32, 32, 3)
        # Main path
        x_main = self.conv1_main(inputs)  # output tensor shape: (1, 32, 32, 32)
        x_main = self.conv2_main(x_main)  # output tensor shape: (1, 32, 32, 64)
        x_main = self.pool_main(x_main)  # output tensor shape: (1, 16, 16, 64)

        # Branch path
        x_branch = self.conv1_branch(inputs)  # output tensor shape of this layer should be (1, 32, 32, 64)
        x_branch = self.pool_branch(x_branch)

        # Combine outputs using addition
        x_combined = self.Add()([x_main, x_branch])

        # Flatten and project to 10 classes
        x_flattened = self.flatten(x_combined)
        x_output = self.dense1(x_flattened)
        x_output = self.dense2(x_output)

        return x_output

if __name__ == "__main__":
    # Dummy input tensor with shape (batch_size, height, width, channels)
    input_tensor = tf.random.normal(shape=(1, 32, 32, 3))
    model = DL_Model()
    output_tensor = model(input_tensor)
    print("Input shape:", input_tensor.shape)
    print("Output shape:", output_tensor.shape)