import tensorflow as tf

class DL_Model:
    def __init__(self):
        self.model = tf.keras.Sequential()

    def __call__(self, x):
        x = self.model(x)
        return x

    def build(self):
        self.model.add(tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu'))
        self.model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))
        self.model.add(tf.keras.layers.Flatten())
        self.model.add(tf.keras.layers.Dense(10, activation='softmax'))
        return self.model  # Return the model object

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal((1, 32, 32, 3))
    output_tensor = model.build()(input_tensor)