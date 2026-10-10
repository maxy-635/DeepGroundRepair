import tensorflow as tf

class DL_Model(tf.keras.Model):
    def __init__(self):
        super(DL_Model, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(32, (3, 3), strides=(1, 1), padding='same')
        self.branch1 = tf.keras.layers.Conv2D(32, (3, 3), strides=(1, 1), padding='same')
        self.branch2 = tf.keras.layers.Conv2D(64, (3, 3), strides=(1, 1), padding='same')
        self.branch3 = tf.keras.layers.Conv2D(64, (3, 3), strides=(1, 1), padding='same')
        self.max_pool2d = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid')
        self.conv2_transpose1 = tf.keras.layers.Conv2DTranspose(64, (3, 3), strides=(1, 1), padding='same')
        self.conv2_transpose2 = tf.keras.layers.Conv2DTranspose(32, (3, 3), strides=(1, 1), padding='same')
        self.concatenate = tf.keras.layers.Concatenate(axis=-1)
        self.conv_concat = tf.keras.layers.Conv2D(64, (1, 1), strides=(1, 1), padding='same')
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(128, activation=tf.keras.activations.relu)
        self.dense2 = tf.keras.layers.Dense(10, activation=tf.keras.activations.softmax)

    def call(self, inputs):
        main_path = self.conv1(inputs)
        branch1 = self.branch1(main_path)
        branch2 = self.max_pool2d(main_path)
        branch2 = self.branch2(branch2)
        branch2 = self.conv2_transpose1(branch2)
        branch3 = self.max_pool2d(main_path)
        branch3 = self.branch3(branch3)
        branch3 = self.conv2_transpose2(branch3)
        concat = self.concatenate([branch1, branch2, branch3])
        concat = self.conv_concat(concat)
        flattened = self.flatten(concat)
        output = self.dense1(flattened)
        output = self.dense2(output)
        return output

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal((1, 32, 32, 3))
    output_tensor = model(input_tensor)
    print(output_tensor.shape)