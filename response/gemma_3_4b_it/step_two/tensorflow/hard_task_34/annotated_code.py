import tensorflow as tf

class DL_Model(tf.keras.Model):
    def __init__(self):
        super(DL_Model, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(
            filters=32,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='same',
            activation=tf.keras.layers.ReLU()
        )
        self.conv2 = tf.keras.layers.Conv2D(
            filters=32,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='same',
            activation=tf.keras.layers.ReLU()
        )
        self.conv3 = tf.keras.layers.Conv2D(
            filters=32,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='same',
            activation=tf.keras.layers.ReLU()
        )
        self.conv4 = tf.keras.layers.Conv2D(filters='mask', kernel_size=(3, 3), strides=(1, 1), padding='same', activation=tf.keras.layers.ReLU())






        self.conv5 = tf.keras.layers.Conv2D(
            filters=10,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='same',
            activation=None
        )
        self.dense = tf.keras.layers.Dense(10, activation='softmax')

    def call(self, inputs):
        # the initial shape of inputs is: (1, 28, 28, 1)
        main_path = inputs  # output tensor shape: (1, 28, 28, 1)
        main_path = self.conv1(main_path)  # output tensor shape: (1, 28, 28, 32)
        main_path = self.conv2(main_path)  # output tensor shape: (1, 28, 28, 32)
        main_path = self.conv3(main_path)  # output tensor shape: (1, 28, 28, 32)
        main_path = self.conv4(main_path)  # output tensor shape of this layer should be (1, 28, 28, 10)
        branch_path = self.conv5(main_path)  # output tensor shape: (1, 28, 28, 10)
        merged = tf.keras.layers.Add()([main_path, branch_path])
        flattened = tf.keras.layers.Flatten()(merged)
        output = self.dense(flattened)
        return output

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal((1, 28, 28, 1))
    output_tensor = model(input_tensor)
    print(output_tensor.shape)