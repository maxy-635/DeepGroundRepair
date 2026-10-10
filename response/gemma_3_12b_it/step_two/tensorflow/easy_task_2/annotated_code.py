import tensorflow as tf
from tensorflow.keras.layers import Conv2D, AveragePooling2D, Flatten, Dense, Dropout, Softmax
from tensorflow.keras.models import Sequential


class DL_Model(tf.keras.Model):
    def __init__(self):
        super(DL_Model, self).__init__()
        # Define the layers
        self.conv1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')
        self.pool1 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')

        self.conv2 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')
        self.pool2 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')

        self.conv3 = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')
        self.conv4 = Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')
        self.pool3 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')

        self.flatten = Flatten()
        self.dense1 = Dense(units=1024, activation='relu')
        self.dropout1 = Dropout(rate=0.5)
        self.dense2 = Dense(units=512, activation='relu')
        self.dropout2 = Dropout(rate=0.5)
        self.softmax = Softmax(axis=-1)

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.conv4(x)
        x = self.pool3(x)

        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dropout1(x)
        x = self.dense2(x)
        x = self.dropout2(x)
        x = self.softmax(x)
        return x


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal((1, 224, 224, 3))  # Dummy input tensor
    output_tensor = model(input_tensor)
    print(output_tensor.shape)