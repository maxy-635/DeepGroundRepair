import tensorflow as tf
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Reshape, Multiply, Conv2D, MaxPool2D, Add, Flatten
from tensorflow.keras.models import Sequential

class DL_Model(Sequential):
    def __init__(self, **kwargs):
        super(DL_Model, self).__init__(**kwargs)
        self.add(GlobalAveragePooling2D(keepdims=True))
        self.add(Dense(units=64, activation='relu'))
        self.add(Dense(units=64, activation='relu'))
        self.add(Reshape(target_shape=(1, 1, 64)))
        self.add(Multiply())
        self.add(Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'))
        self.add(Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'))
        self.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))
        self.add(Add())
        self.add(Flatten())
        self.add(Dense(units=128, activation='relu'))
        self.add(Dense(units=10, activation='softmax'))

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.keras.Input(shape=(32, 32, 3))
    output_tensor = model(input_tensor)
    model = tf.keras.Model(inputs=input_tensor, outputs=output_tensor)
    model.summary()