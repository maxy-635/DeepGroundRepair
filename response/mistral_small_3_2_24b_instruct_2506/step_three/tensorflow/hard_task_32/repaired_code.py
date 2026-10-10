import tensorflow as tf
from tensorflow.keras.layers import DepthwiseConv2D, Conv2D, Dropout, Concatenate, Dense
from tensorflow.keras.models import Sequential

class DL_Model(Sequential):
    def __init__(self):
        super(DL_Model, self).__init__()
        self.build_model()

    def build_model(self):
        # First branch
        branch1 = Sequential([
            DepthwiseConv2D(kernel_size=3, strides=1, padding='same', depth_multiplier=1, activation='relu'),
            Conv2D(filters=32, kernel_size=1, strides=1, padding='same', activation='relu'),
            Dropout(rate=0.5)
        ])

        # Second branch
        branch2 = Sequential([
            DepthwiseConv2D(kernel_size=3, strides=1, padding='same', depth_multiplier=1, activation='relu'),
            Conv2D(filters=32, kernel_size=1, strides=1, padding='same', activation='relu'),
            Dropout(rate=0.5)
        ])

        # Third branch
        branch3 = Sequential([
            DepthwiseConv2D(kernel_size=3, strides=1, padding='same', depth_multiplier=1, activation='relu'),
            Conv2D(filters=32, kernel_size=1, strides=1, padding='same', activation='relu'),
            Dropout(rate=0.5)
        ])

        # Concatenate branches
        self.add(Concatenate(axis=-1)([branch1.output, branch2.output, branch3.output]))

        # Fully connected layers
        self.add(Dense(units=128, activation='relu'))
        self.add(Dense(units=10, activation='softmax'))

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.keras.Input(shape=(28, 28, 1))
    output_tensor = model(input_tensor)
    model = tf.keras.Model(inputs=input_tensor, outputs=output_tensor)
    model.summary()