import tensorflow as tf

class DL_Model(tf.keras.Model):
    def __init__(self):
        super(DL_Model, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(
            filters=32,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='same',
            activation='relu'
        )
        self.conv2 = tf.keras.layers.Conv2D(
            filters=10,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='same',
            activation='relu'
        )
        self.branch = tf.keras.layers.Dense(10)
        self.add = tf.keras.layers.Add()
        self.flatten = tf.keras.layers.Flatten()
        self.dense = tf.keras.layers.Dense(10, activation='softmax')

    def call(self, inputs):
        main_path = self.conv1(inputs)
        main_path = self.conv2(main_path)
        branch_path = self.branch(inputs)
        combined = self.add([main_path, branch_path])
        flattened = self.flatten(combined)
        output = self.dense(flattened)
        return output

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal((1, 28, 28, 1))
    output_tensor = model(input_tensor)
    print(output_tensor.shape)