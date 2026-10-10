import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Concatenate, Dense

class DL_Model(tf.keras.Model):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Convolutional and Pooling Layers
        self.conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')
        self.pool1 = MaxPool2D(pool_size=(1, 1), strides=(1, 1), padding='valid')

        self.conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')
        self.pool2 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid')

        self.conv3 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')
        self.pool3 = MaxPool2D(pool_size=(4, 4), strides=(4, 4), padding='valid')

        # Flatten and Concatenate Layers
        self.flatten1 = Flatten()
        self.flatten2 = Flatten()
        self.flatten3 = Flatten()
        self.concat = Concatenate()

        # Fully Connected Layers
        self.dense1 = Dense(units=256, activation='relu')
        self.dense2 = Dense(units=10, activation='softmax')

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.pool3(x)

        x1 = self.flatten1(x)
        x2 = self.flatten2(x)
        x3 = self.flatten3(x)

        x = self.concat([x1, x2, x3])

        x = self.dense1(x)
        output = self.dense2(x)
        return output


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal((1, 32, 32, 3))  # Example input shape (batch_size, height, width, channels)
    output_tensor = model(input_tensor)
    print("Output shape:", output_tensor.shape)