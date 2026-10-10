import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Reshape, Multiply, Flatten

class DL_Model(Sequential):
    def __init__(self):
        super(DL_Model, self).__init__()
        self.add(GlobalAveragePooling2D(keepdims=False))
        self.add(Dense(units=64, activation='relu'))
        self.add(Dense(units=64, activation='relu'))
        self.add(Reshape(target_shape=(1, 1, 64)))
        self.add(Multiply())
        self.add(Flatten())
        self.add(Dense(units=10, activation='softmax'))

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.keras.Input(shape=(32, 32, 3))
    output_tensor = model(input_tensor)
    model = tf.keras.Model(inputs=input_tensor, outputs=output_tensor)
    model.summary()