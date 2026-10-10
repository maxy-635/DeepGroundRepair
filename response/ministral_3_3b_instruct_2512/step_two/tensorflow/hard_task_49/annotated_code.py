import tensorflow as tf

class DL_Model:
    def __init__(self):
        self.model = tf.keras.Sequential()

    def __call__(self, x):
        x = self.model(x)
        return x

    def build(self):
        self.model.add(tf.keras.layers.Input(shape=(28, 28, 1)))
        self.model.add(tf.keras.layers.AveragePooling2D(pool_size=(1, 1), strides=(1, 1), padding='valid'))
        self.model.add(tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))
        self.model.add(tf.keras.layers.AveragePooling2D(pool_size=(4, 4), strides=(4, 4), padding='valid'))
        self.model.add(tf.keras.layers.Flatten())
        self.model.add(tf.keras.layers.Dense(units=10, activation='softmax'))

        # First block output shape: (batch_size, 1, 1, 3)
        # Reshape to (batch_size, 4, 1, 1)
        x = self.model.layers[-2].output
        x = tf.keras.layers.Reshape(target_shape=(4, 1, 1))(x)

        # Second block
        self.model.add(tf.keras.layers.DepthwiseConv2D(kernel_size=(1, 1), strides=(1, 1), padding='valid', depth_multiplier=1, activation='relu'))
        self.model.add(tf.keras.layers.DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='valid', depth_multiplier=1, activation='relu'))
        self.model.add(tf.keras.layers.DepthwiseConv2D(kernel_size=(5, 5), strides=(1, 1), padding='valid', depth_multiplier=1, activation='relu'))
        self.model.add(tf.keras.layers.DepthwiseConv2D(kernel_size=(7, 7), strides=(1, 1), padding='valid', depth_multiplier=1, activation='relu'))
        self.model.add(tf.keras.layers.Concatenate(axis=-1))
        self.model.add(tf.keras.layers.Flatten())
        self.model.add(tf.keras.layers.Dense(units=10, activation='softmax'))

        return self.model

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal(shape=(1, 28, 28, 1))
    output_tensor = model(input_tensor)
    print(output_tensor.shape)