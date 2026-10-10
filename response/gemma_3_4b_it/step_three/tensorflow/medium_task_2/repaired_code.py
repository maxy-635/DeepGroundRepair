import tensorflow as tf

class DL_Model(tf.keras.Model):
    def __init__(self):
        super(DL_Model, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=1, padding='same')
        self.conv2 = tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=1, padding='same')
        self.max_pool = tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding='valid')
        self.conv_branch = tf.keras.layers.Conv2D(filters=64, kernel_size=5, strides=1, padding='same')
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(units=128, activation='relu')
        self.dense2 = tf.keras.layers.Dense(units=10, activation='softmax')

    def call(self, inputs):
        # the initial shape of inputs is: (1, 32, 32, 3)
        main_path = self.conv1(inputs)  # output tensor shape: (1, 32, 32, 32)
        main_path = self.conv2(main_path)  # output tensor shape: (1, 32, 32, 64)
        main_path = self.max_pool(main_path)  # output tensor shape: (1, 16, 16, 64)

        branch_path = self.conv_branch(inputs)  # output tensor shape: (1, 32, 32, 64)
        branch_path = self.flatten(branch_path)  # output tensor shape: (1, 65536)

        combined = tf.keras.layers.concatenate([main_path, branch_path], axis=-1)
        combined = self.dense1(combined)
        outputs = self.dense2(combined)
        return outputs

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal((1, 32, 32, 3))
    output_tensor = model(input_tensor)
    print(output_tensor.shape)