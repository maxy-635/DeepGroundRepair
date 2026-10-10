import tensorflow as tf
from tensorflow.keras.layers import Conv2D, AveragePooling2D, Flatten, Dense, Dropout
from tensorflow.keras.models import Sequential

class DL_Model(Sequential):
    def __init__(self):
        super(DL_Model, self).__init__()
        self.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(224, 224, 3)))
        self.add(AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))
        self.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
        self.add(AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))
        self.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu'))
        self.add(Conv2D(filters=256, kernel_size=(3, 3), activation='relu'))
        self.add(Conv2D(filters=512, kernel_size=(3, 3), activation='relu'))
        self.add(AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))
        self.add(Flatten())
        self.add(Dense(units=1024, activation='relu'))
        self.add(Dropout(rate=0.5))
        self.add(Dense(units=1024, activation='relu'))
        self.add(Dropout(rate=0.5))
        self.add(Dense(units=1000, activation='softmax'))

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.keras.Input(shape=(224, 224, 3))
    output_tensor = model(input_tensor)
    model.summary()