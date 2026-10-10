import tensorflow as tf

class DL_Model(tf.keras.Model):
    def __init__(self):
        super(DL_Model, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(32, (1, 1), strides=1, padding='same')
        self.conv2 = tf.keras.layers.Conv2D(32, (3, 3), strides=1, padding='same')
        self.conv3 = tf.keras.layers.Conv2D(32, (3, 3), strides=1, padding='same')
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(128, activation=tf.nn.relu)
        self.dense2 = tf.keras.layers.Dense(64, activation=tf.nn.relu)
        self.output_layer = tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)

    def call(self, inputs):
        main_path = self.conv1(inputs)
        main_path = self.conv2(main_path)
        main_path = self.conv3(main_path)

        branch_path = tf.keras.layers.Conv2D(32, (3, 3), strides=1, padding='same')(inputs)
        branch_path = tf.nn.relu(branch_path)

        merged = tf.keras.layers.Add()([main_path, branch_path])
        flattened = self.flatten(merged)
        dense1 = self.dense1(flattened)
        dense2 = self.dense2(dense1)
        output = self.output_layer(dense2)

        return output

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal((32, 32, 64))
    output_tensor = model(input_tensor)
    print(output_tensor.shape)