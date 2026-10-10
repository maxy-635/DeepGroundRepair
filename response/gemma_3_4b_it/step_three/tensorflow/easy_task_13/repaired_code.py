import tensorflow as tf

class DL_Model(tf.keras.Model):
    def __init__(self):
        super(DL_Model, self).__init__()
        # First 1x1 convolutional layer
        self.conv1 = tf.keras.layers.Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same')
        self.dropout1 = tf.keras.layers.Dropout(rate=0.2)
        # Second 1x1 convolutional layer
        self.conv2 = tf.keras.layers.Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same')
        self.dropout2 = tf.keras.layers.Dropout(rate=0.2)
        # 3x1 convolutional layer
        self.conv3 = tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 1), strides=(1, 1), padding='same')
        self.dropout3 = tf.keras.layers.Dropout(rate=0.2)
        # 1x3 convolutional layer
        self.conv4 = tf.keras.layers.Conv2D(filters=256, kernel_size=(1, 3), strides=(1, 1), padding='same')
        self.dropout4 = tf.keras.layers.Dropout(rate=0.2)
        # 1x1 convolutional layer to restore channels
        self.conv5 = tf.keras.layers.Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same')
        self.batch_norm = tf.keras.layers.BatchNormalization()
        # Flatten layer
        self.flatten = tf.keras.layers.Flatten()
        # Dense layer
        self.dense = tf.keras.layers.Dense(units=10, activation='softmax')

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.dropout1(x)
        x = self.conv2(x)
        x = self.dropout2(x)
        x = self.conv3(x)
        x = self.dropout3(x)
        x = self.conv4(x)
        x = self.dropout4(x)
        x = self.conv5(x)
        x = self.batch_norm(x)
        x = tf.add(inputs, x)
        x = self.flatten(x)
        x = self.dense(x)
        return x


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal((1, 28, 28, 1))  # Example input shape: (batch_size, height, width, channels)
    output_tensor = model(input_tensor)
    print(output_tensor.shape)