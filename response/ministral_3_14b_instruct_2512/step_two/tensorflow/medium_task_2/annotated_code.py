import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense, concatenate
from tensorflow.keras.models import Model

class DL_Model(Model):
    def __init__(self):
        super(DL_Model, self).__init__()
        # Main path layers
        self.conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1),
                           padding='same', activation='relu')
        self.conv2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1),
                           padding='same', activation='relu')
        self.pool = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid')

        # Branch path layer
        self.conv_branch = Conv2D(filters=32, kernel_size='mask', strides='mask', padding='mask', activation='relu'),


        # Fully connected layers
        self.flatten = Flatten()
        self.dense1 = Dense(units=128, activation='relu')
        self.dense2 = Dense(units=10, activation='softmax')

    def call(self, inputs):
        # the initial shape of inputs is: (1, 32, 32, 3)
        # Main path
        x = self.conv1(inputs)  # output tensor shape: (1, 32, 32, 32)
        x = self.conv2(x)  # output tensor shape: (1, 32, 32, 32)
        x = self.pool(x)  # output tensor shape: (1, 16, 16, 32)

        # Branch path
        branch = self.conv_branch(inputs)  # output tensor shape of this layer should be (1, 16, 16, 32)

        # Combine paths
        combined = concatenate([x, branch], axis=-1)

        # Fully connected layers
        x = self.flatten(combined)
        x = self.dense1(x)
        x = self.dense2(x)

        return x

if __name__ == "__main__":
    model = DL_Model()
    # CIFAR-10 images are 32x32 with 3 channels
    input_tensor = tf.random.normal(shape=(1, 32, 32, 3))
    output_tensor = model(input_tensor)
    print("Input shape:", input_tensor.shape)
    print("Output shape:", output_tensor.shape)