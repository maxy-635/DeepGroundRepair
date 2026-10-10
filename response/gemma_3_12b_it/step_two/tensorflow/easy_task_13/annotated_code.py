import tensorflow as tf

class DL_Model(tf.keras.Model):
    def __init__(self):
        super(DL_Model, self).__init__()

        self.conv1_1 = tf.keras.layers.Conv1D(filters=32, kernel_size=1, strides=1, padding='same', activation='relu')
        self.dropout1 = tf.keras.layers.Dropout(rate=0.2)
        self.conv3_1 = tf.keras.layers.Conv1D(filters=64, kernel_size=3, strides=1, padding='same', activation='relu')
        self.dropout2 = tf.keras.layers.Dropout(rate=0.2)
        self.conv1_3 = tf.keras.layers.Conv1D(filters=32, kernel_size=1, strides=1, padding='same', activation='relu')
        self.dropout3 = tf.keras.layers.Dropout(rate=0.2)
        self.conv1_1_restore = tf.keras.layers.Conv1D(filters=28, kernel_size=1, strides=1, padding='same', activation='relu')
        self.flatten = tf.keras.layers.Flatten()
        self.dense = tf.keras.layers.Dense(units=10, activation='softmax')

    def call(self, inputs):
        x = self.conv1_1(inputs)
        x = self.dropout1(x)
        x = self.conv3_1(x)
        x = self.dropout2(x)
        x = self.conv1_3(x)
        x = self.dropout3(x)
        x = self.conv1_1_restore(x)
        
        x = tf.add(inputs, x)
        x = self.flatten(x)
        x = self.dense(x)
        return x


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal(shape=(1, 28, 28))  # Example input shape for MNIST
    output_tensor = model(input_tensor)
    print(output_tensor.shape)