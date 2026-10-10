import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Add, Flatten, Dense

class DL_Model:
    def __init__(self):
        self.model = tf.keras.Sequential()

        # Main path
        self.model.add(Conv2D(32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'))
        self.model.add(Conv2D(1, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'))

        # Branch path - identity connection
        self.model.add(Conv2D(1, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu'))

        # Addition operation
        self.model.add(Add())

        # Flatten and Dense layer
        self.model.add(Flatten())
        self.model.add(Dense(10, activation='softmax'))

    def __call__(self, input_tensor):
        return self.model(input_tensor)

# For functional API approach (better for this architecture)
from tensorflow.keras.layers import Input

class DL_Model:
    def __init__(self):
        # Input layer
        input_layer = Input(shape=(28, 28, 1))

        # Main path
        x = Conv2D(32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
        x = Conv2D(1, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(x)

        # Branch path
        branch = Conv2D(1, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)

        # Addition operation
        x = Add()([x, branch])

        # Flatten and Dense layer
        x = Flatten()(x)
        output = Dense(10, activation='softmax')(x)

        self.model = tf.keras.Model(inputs=input_layer, outputs=output)

    def __call__(self, input_tensor):
        return self.model(input_tensor)

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal((1, 28, 28, 1))  # MNIST input shape
    output_tensor = model(input_tensor)
    print(output_tensor.shape)