import tensorflow as tf
from tensorflow.keras.layers import Conv2D, AveragePooling2D, Concatenate, Dense, Flatten
from tensorflow.keras.models import Sequential

class DL_Model(Sequential):
    def __init__(self):
        super(DL_Model, self).__init__()
        self.add(Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu'))  # Path 1
        self.add(AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))  # Path 2
        self.add(Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu'))

        # Path 3
        conv1x3 = Conv2D(filters=64, kernel_size=(1, 3), strides=(1, 1), padding='same', activation='relu')
        conv3x1 = Conv2D(filters=64, kernel_size=(3, 1), strides=(1, 1), padding='same', activation='relu')
        self.add(Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu'))
        self.add(conv1x3)
        self.add(conv3x1)
        self.add(Concatenate(axis=-1)([conv1x3, conv3x1]))

        # Path 4
        conv3x3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')
        conv1x3_2 = Conv2D(filters=64, kernel_size=(1, 3), strides=(1, 1), padding='same', activation='relu')
        conv3x1_2 = Conv2D(filters=64, kernel_size=(3, 1), strides=(1, 1), padding='same', activation='relu')
        self.add(Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu'))
        self.add(conv3x3)
        self.add(conv1x3_2)
        self.add(conv3x1_2)
        self.add(Concatenate(axis=-1)([conv1x3_2, conv3x1_2]))

        # Concatenate all paths
        self.add(Concatenate(axis=-1))
        self.add(Flatten())
        self.add(Dense(units=10, activation='softmax'))

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.keras.Input(shape=(32, 32, 3))
    output_tensor = model(input_tensor)
    model.summary()