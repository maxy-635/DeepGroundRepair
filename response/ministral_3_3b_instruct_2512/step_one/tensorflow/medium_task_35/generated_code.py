import tensorflow as tf
from tensorflow.keras import layers

class DL_Model(tf.keras.Model):
    def __init__(self):
        super(DL_Model, self).__init__()
        self.conv1 = layers.Conv2D(32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')
        self.pool1 = layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid')
        self.conv2 = layers.Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')
        self.dropout1 = layers.Dropout(0.25)

        self.upconv1 = layers.Conv2D(64, kernel_size=(2, 2), strides=(2, 2), padding='same', activation='relu')
        self.conv3 = layers.Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')
        self.conv4 = layers.Conv2D(32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')

        self.upconv2 = layers.Conv2D(32, kernel_size=(2, 2), strides=(2, 2), padding='same', activation='relu')
        self.conv5 = layers.Conv2D(32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')

        self.final_conv = layers.Conv2D(10, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='softmax')

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.dropout1(x)

        # Upsampling path
        skip_connection1 = x
        x = self.upconv1(x)
        x = tf.keras.layers.Concatenate()([x, skip_connection1])
        x = self.conv3(x)
        x = self.conv4(x)

        skip_connection2 = x
        x = self.upconv2(x)
        x = tf.keras.layers.Concatenate()([x, skip_connection2])
        x = self.conv5(x)

        x = self.final_conv(x)
        return x

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal(shape=(1, 32, 32, 3))
    output_tensor = model(input_tensor)
    print(output_tensor.shape)