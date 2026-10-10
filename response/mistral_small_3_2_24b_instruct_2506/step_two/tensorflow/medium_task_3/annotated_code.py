import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, InputLayer
from tensorflow.keras.models import Sequential

class DL_Model(Sequential):
    def __init__(self):
        super().__init__()
        self.add(InputLayer(input_shape=(28, 28, 1)))
        # First sequential block
        self.add(Conv2D(filters=1, kernel_size=(3, 3), strides=(1, 1), padding='same'))
        self.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))
        # Second sequential block
        self.add(Conv2D(filters=1, kernel_size=(3, 3), strides=(1, 1), padding='same'))
        self.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))
        # Addition operation with input
        self.add(tf.keras.layers.Lambda(lambda x: tf.math.add(x[0], x[1])))
        # Fully connected layer for classification
        self.add(Dense(units=10, activation='softmax'))

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal(shape=(1, 28, 28, 1))
    output_tensor = model(input_tensor)
    print(output_tensor)