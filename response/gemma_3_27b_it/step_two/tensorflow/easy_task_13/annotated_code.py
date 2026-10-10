import tensorflow as tf

class DL_Model(tf.keras.Model):
    def __init__(self):
        super(DL_Model, self).__init__()
        # Define the layers
        self.conv1x1_1 = tf.keras.layers.Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')
        self.dropout1 = tf.keras.layers.Dropout(rate=0.2)
        self.conv1x1_2 = tf.keras.layers.Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')
        self.dropout2 = tf.keras.layers.Dropout(rate=0.2)
        self.conv3x1 = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 1), strides=(1, 1), padding='same', activation='relu')
        self.dropout3 = tf.keras.layers.Dropout(rate=0.2)
        self.conv1x3 = tf.keras.layers.Conv2D(filters=32, kernel_size=(1, 3), strides=(1, 1), padding='same', activation='relu')
        self.dropout4 = tf.keras.layers.Dropout(rate=0.2)
        self.conv1x1_restore = tf.keras.layers.Conv2D(filters=1, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')
        self.dropout5 = tf.keras.layers.Dropout(rate=0.2)
        self.flatten = tf.keras.layers.Flatten()
        self.dense = tf.keras.layers.Dense(units=10, activation='softmax')

    def call(self, inputs):
        # Perform the operations
        x = self.conv1x1_1(inputs)
        x = self.dropout1(x)
        x = self.conv1x1_2(x)
        x = self.dropout2(x)
        x = self.conv3x1(x)
        x = self.dropout3(x)
        x = self.conv1x3(x)
        x = self.dropout4(x)
        x = self.conv1x1_restore(x)
        x = self.dropout5(x)
        x = tf.keras.layers.Add()([x, inputs])
        x = self.flatten(x)
        x = self.dense(x)
        return x


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal(shape=(1, 28, 28, 1))
    output_tensor = model(input_tensor)
    print(output_tensor.shape)