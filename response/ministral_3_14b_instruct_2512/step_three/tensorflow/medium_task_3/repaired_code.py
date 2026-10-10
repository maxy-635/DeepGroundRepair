import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPool2D, Add, Dense, Flatten, concatenate, Reshape
from tensorflow.keras.models import Model

class DL_Model(Model):
    def __init__(self):
        super(DL_Model, self).__init__()
        # First block: Conv2D -> MaxPool2D
        self.conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')
        self.maxpool1 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid')

        # Second block: Conv2D -> MaxPool2D
        self.conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')
        self.maxpool2 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid')

        # Final Conv2D to match input channel dimension
        self.conv_final = Conv2D(filters=1, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')

        # Fully connected layer for classification
        self.flatten = Flatten()
        self.dense = Dense(units=10, activation='softmax')

    def call(self, inputs):
        # First block
        x1 = self.conv1(inputs)
        x1 = self.maxpool1(x1)

        # Second block
        x2 = self.conv2(x1)
        x2 = self.maxpool2(x2)

        # Final Conv2D to match input channel dimension
        x_final = self.conv_final(x2)

        # Combine with input through addition
        combined = Add()([inputs, x_final])

        # Flatten and classify
        flattened = self.flatten(combined)
        output = self.dense(flattened)

        return output

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal(shape=(1, 28, 28, 1))  # MNIST images are 28x28 with 1 channel
    output_tensor = model(input_tensor)
    print("Input shape:", input_tensor.shape)
    print("Output shape:", output_tensor.shape)