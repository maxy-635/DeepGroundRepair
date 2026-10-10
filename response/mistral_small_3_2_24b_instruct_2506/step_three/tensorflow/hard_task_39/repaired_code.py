import tensorflow as tf
from tensorflow.keras.layers import MaxPool2D, Flatten, Concatenate, Dense, Reshape, Conv2D
from tensorflow.keras.models import Sequential

class DL_Model(Sequential):
    def __init__(self):
        super(DL_Model, self).__init__()
        # Block 1
        self.add(MaxPool2D(pool_size=(1, 1), strides=(1, 1), padding='valid'))
        self.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))
        self.add(MaxPool2D(pool_size=(4, 4), strides=(4, 4), padding='valid'))
        self.add(Flatten())
        self.add(Flatten())
        self.add(Flatten())
        self.add(Reshape(target_shape=(-1, 1)))  # Reshape to have at least 2D input for Concatenate
        self.add(Concatenate(axis=1))
        self.add(Dense(units=128, activation='relu'))
        self.add(Reshape(target_shape=(8, 8, 2)))

        # Block 2
        self.add(Conv2D(filters=1, kernel_size=(1, 1), strides=(1, 1), padding='valid', activation='relu'))
        self.add(Conv2D(filters=1, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu'))
        self.add(Conv2D(filters=1, kernel_size=(5, 5), strides=(1, 1), padding='valid', activation='relu'))
        self.add(MaxPool2D(pool_size=(3, 3), strides=(1, 1), padding='valid'))
        self.add(Concatenate(axis=-1))
        self.add(Flatten())
        self.add(Dense(units=10, activation='softmax'))

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.keras.Input(shape=(28, 28, 1))
    output_tensor = model(input_tensor)
    model.build(input_shape=(None, 28, 28, 1))
    model.summary()