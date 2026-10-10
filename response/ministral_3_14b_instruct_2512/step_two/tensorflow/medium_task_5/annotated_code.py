import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPool2D, Add, Flatten, Dense, Input
from tensorflow.keras.models import Model

class DL_Model(Model):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Main path
        self.conv1_main = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')
        self.conv2_main = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')
        self.pool1_main = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid')

        # Branch path
        self.conv1_branch = Conv2D(filters='mask', kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')
        self.pool1_branch = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid')

        # Fully connected layers
        self.flatten = Flatten()
        self.dense1 = Dense(units=128, activation='relu')
        self.dense2 = Dense(units=10, activation='softmax')

    def call(self, inputs):
        # the initial shape of inputs is: (1, 32, 32, 3)
        # Main path
        x_main = self.conv1_main(inputs)  # output tensor shape: (1, 32, 32, 32)
        x_main = self.conv2_main(x_main)  # output tensor shape: (1, 32, 32, 32)
        x_main = self.pool1_main(x_main)  # output tensor shape: (1, 16, 16, 32)

        # Branch path
        x_branch = self.conv1_branch(inputs)  # output tensor shape of this layer should be (1, 32, 32, 32)
        x_branch = self.pool1_branch(x_branch)

        # Combine paths
        combined = Add()([x_main, x_branch])

        # Fully connected layers
        x = self.flatten(combined)
        x = self.dense1(x)
        x = self.dense2(x)

        return x

if __name__ == "__main__":
    model = DL_Model()

    # Create a dummy input tensor with shape (batch_size, height, width, channels)
    # CIFAR-10 images are 32x32 with 3 channels
    input_tensor = tf.random.normal(shape=(1, 32, 32, 3))

    # Get model output
    output_tensor = model(input_tensor)
    print("Input shape:", input_tensor.shape)
    print("Output shape:", output_tensor.shape)