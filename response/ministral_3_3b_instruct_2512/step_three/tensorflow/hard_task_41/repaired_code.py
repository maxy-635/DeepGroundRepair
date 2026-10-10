class DL_Model:
    def __init__(self):
        self.model = tf.keras.Sequential([tf.keras.layers.Input(shape=(28, 28, 1))])

    def __call__(self, x):
        x = self.model(x)
        return x

    def build(self):
        self.model.add(tf.keras.layers.Input(shape=(28, 28, 1)))
        self.model.add(tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu'))
        self.model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
        self.model.add(tf.keras.layers.Flatten())
        self.model.add(tf.keras.layers.Dense(10, activation='softmax'))
        return self.model

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal((1, 28, 28, 1))
    output_tensor = model.build()(input_tensor)