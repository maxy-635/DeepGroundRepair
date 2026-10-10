import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Concatenate, Dense
from tensorflow.keras.models import Model

class DL_Model(Model):
    def __init__(self):
        super(DL_Model, self).__init__()
        # Define the layers
        self.conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')
        self.pool1 = MaxPool2D(pool_size=(1, 1), strides=(1, 1), padding='valid')
        self.pool2 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid')
        self.pool3 = MaxPool2D(pool_size=(4, 4), strides=(4, 4), padding='valid')

        self.flatten1 = Flatten()
        self.flatten2 = Flatten()
        self.flatten3 = Flatten()

        self.concatenate = Concatenate(axis=-1)

        self.dense1 = Dense(units=128, activation='relu')
        self.dense2 = Dense(units=10, activation='softmax')

    def call(self, inputs):
        # First branch with 1x1 max pooling
        x1 = self.conv1(inputs)
        x1 = self.pool1(x1)
        x1 = self.flatten1(x1)

        # Second branch with 2x2 max pooling
        x2 = self.conv1(inputs)
        x2 = self.pool2(x2)
        x2 = self.flatten2(x2)

        # Third branch with 4x4 max pooling
        x3 = self.conv1(inputs)
        x3 = self.pool3(x3)
        x3 = self.flatten3(x3)

        # Concatenate the flattened features
        concatenated = self.concatenate([x1, x2, x3])

        # Fully connected layers
        x = self.dense1(concatenated)
        outputs = self.dense2(x)

        return outputs

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal(shape=(1, 32, 32, 3))  # CIFAR-10 image shape: 32x32x3
    output_tensor = model(input_tensor)
    print("Output shape:", output_tensor.shape)