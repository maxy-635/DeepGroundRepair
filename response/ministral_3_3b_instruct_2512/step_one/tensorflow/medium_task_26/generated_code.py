import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Concatenate, Flatten, Dense, Activation

class DL_Model:
    def __init__(self):
        self.model = tf.keras.Sequential()

    def __call__(self, input_tensor):
        # Compress input channels with 1x1 convolution
        compressed = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='valid', activation=None)(input_tensor)

        # Expand features through two parallel convolutional layers
        conv1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='valid', activation='relu')(compressed)
        conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(compressed)

        # Concatenate the results
        concatenated = Concatenate(axis=-1)([conv1, conv2])

        # Flatten and pass through fully connected layers
        flattened = Flatten()(concatenated)
        dense1 = Dense(units=128, activation='relu')(flattened)
        output = Dense(units=10, activation='softmax')(dense1)  # Assuming 10 classes

        return output

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal((1, 32, 32, 64))
    output_tensor = model(input_tensor)
    print(output_tensor.shape)