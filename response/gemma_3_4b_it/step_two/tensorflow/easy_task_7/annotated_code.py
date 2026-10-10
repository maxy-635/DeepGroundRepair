import tensorflow as tf

class DL_Model(tf.keras.Model):
    def __init__(self):
        super(DL_Model, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(
            filters=32,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='same',
            activation=tf.nn.relu
        )
        self.dropout1 = tf.keras.layers.Dropout(
            rate=0.25
        )
        self.conv2 = tf.keras.layers.Conv2D(
            filters=64,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='same',
            activation=tf.nn.relu
        )
        self.dropout2 = tf.keras.layers.Dropout(
            rate=0.5
        )
        self.conv3 = tf.keras.layers.Conv2D(
            filters=64,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='same',
            activation=tf.nn.relu
        )
        self.flatten = tf.keras.layers.Flatten()
        self.dense = tf.keras.layers.Dense(
            units=10,
            activation='softmax'
        )

    def call(self, inputs):
        main_path = inputs
        main_path = self.conv1(main_path)
        main_path = self.dropout1(main_path)
        main_path = self.conv2(main_path)
        main_path = self.dropout2(main_path)
        main_path = self.conv3(main_path)

        branch_path = inputs
        branch_path = self.flatten(branch_path)
        branch_path = self.dense(branch_path)

        combined = tf.keras.layers.Add()([main_path, branch_path])
        return combined


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.keras.Input(shape=(28, 28, 1))
    output_tensor = model(input_tensor)
    print(output_tensor)