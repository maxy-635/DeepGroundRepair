import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Concatenate, Dense, Add, Flatten
from tensorflow.keras.models import Sequential

class DL_Model(Sequential):
    def __init__(self):
        super(DL_Model, self).__init__()
        self.add(Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu', name='conv2d'))
        self.add(Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu', name='conv2d_1'))
        self.add(Conv2D(filters=64, kernel_size=(1, 3), strides=(1, 1), padding='same', activation='relu', name='conv2d_2'))
        self.add(Conv2D(filters=64, kernel_size=(3, 1), strides=(1, 1), padding='same', activation='relu', name='conv2d_3'))
        self.add(Concatenate(axis=-1, name='concatenate'))
        self.add(Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu', name='conv2d_4'))
        self.add(Add(name='add'))
        self.add(Flatten(name='flatten'))
        self.add(Dense(units=1024, activation='relu', name='dense'))
        self.add(Dense(units=10, activation='softmax', name='dense_1'))

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.keras.Input(shape=(32, 32, 3))
    output_tensor = model(input_tensor)
    model = tf.keras.Model(inputs=input_tensor, outputs=output_tensor)
    model.summary()