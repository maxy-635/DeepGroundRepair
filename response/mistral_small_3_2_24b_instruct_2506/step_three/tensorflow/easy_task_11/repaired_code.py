import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import AveragePooling2D, Conv2D, Flatten, Dense, Dropout

class DL_Model(Sequential):
    def __init__(self):
        super().__init__()
        self.add(AveragePooling2D(pool_size=(5, 5), strides=(3, 3), padding='valid'))
        self.add(Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='valid', activation='relu'))
        self.add(Flatten())
        self.add(Dense(units=128, activation='relu'))
        self.add(Dropout(rate=0.5))
        self.add(Dense(units=10, activation='softmax'))

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.keras.Input(shape=(28, 28, 1))
    output_tensor = model(input_tensor)
    model = tf.keras.Model(inputs=input_tensor, outputs=output_tensor)
    model.summary()