import tensorflow as tf

class DL_Model:
    def __init__(self):
        self.model = tf.keras.Sequential([
            tf.keras.layers.DepthwiseConv2D(
                kernel_size=3,
                strides=(2, 2),
                padding='same',
                depth_multiplier=16
            ),
            tf.keras.layers.Activation('relu'),
            tf.keras.layers.Dropout(rate=0.25),
            tf.keras.layers.Conv2D(
                filters=32,
                kernel_size=3,
                strides=(1, 1),
                padding='same',
                activation='relu'
            ),
            tf.keras.layers.Dropout(rate=0.25),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(
                units=10,
                activation='softmax'
            )
        ])

    def __call__(self, input_tensor):
        return self.model(input_tensor)


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal(shape=(1, 28, 28, 1))
    output_tensor = model(input_tensor)
    print(output_tensor.shape)