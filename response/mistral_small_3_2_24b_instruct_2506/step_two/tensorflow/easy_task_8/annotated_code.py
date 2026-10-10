import tensorflow as tf
from tensorflow.keras.layers import DepthwiseConv2D, Conv2D, Dropout, Flatten, Dense
from tensorflow.keras.models import Sequential

class DL_Model(Sequential):
    def __init__(self):
        super(DL_Model, self).__init__()
        self.add(DepthwiseConv2D(kernel_size=3, strides=1, padding='same', depth_multiplier=1, activation='relu'))
        self.add(Conv2D(filters=32, kernel_size=3, strides=1, padding='same', activation='relu'))
        self.add(Dropout(rate=0.25))
        self.add(Conv2D(filters=64, kernel_size=3, strides=1, padding='same', activation='relu'))
        self.add(Dropout(rate=0.25))
        self.add(Flatten())
        self.add(Dropout(rate=0.5))
        self.add(Dense(units=10, activation='softmax'))

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.keras.Input(shape=(28, 28, 1))
    output_tensor = model(input_tensor)
    model = tf.keras.Model(inputs=input_tensor, outputs=output_tensor)
    model.summary()