import tensorflow as tf

class DL_Model(tf.keras.Model):
    def __init__(self):
        super(DL_Model, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(filters=32, kernel_size=(7, 7), strides=(1, 1), padding='valid', activation=None)
        self.ln1 = tf.keras.layers.LayerNormalization(axis=-1, epsilon=1e-5, center=True, scale=True)
        self.conv2 = tf.keras.layers.Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='valid', activation=None)
        self.conv3 = tf.keras.layers.Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='valid', activation=None)
        self.add = tf.keras.layers.Add()
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(units=128, activation='relu')
        self.dense2 = tf.keras.layers.Dense(units=10, activation='softmax')

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.ln1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        branch = inputs
        x = self.add([x, branch])
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dense2(x)
        return x

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal((1, 32, 32, 3))  # Example input shape for CIFAR-10 (batch_size=1, height=32, width=32, channels=3)
    output_tensor = model(input_tensor)
    print(output_tensor.shape)