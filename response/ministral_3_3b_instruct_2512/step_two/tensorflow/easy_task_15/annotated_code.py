import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPool2D, AveragePooling2D, Dropout, Flatten, Dense, GlobalAveragePooling2D

class DL_Model(tf.keras.Model):
    def __init__(self):
        super(DL_Model, self).__init__()
        self.conv_block1 = tf.keras.Sequential([
            Conv2D(32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'),
            Conv2D(64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')
        ])
        self.pool1 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')
        self.dropout1 = Dropout(rate=0.25)

        self.conv_block2 = tf.keras.Sequential([
            Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'),
            Conv2D(128, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')
        ])
        self.pool2 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')
        self.dropout2 = Dropout(rate=0.25)

        self.global_avg_pool = GlobalAveragePooling2D(keepdims=False)
        self.flatten = Flatten()
        self.fc = Dense(10, activation='softmax')

    def call(self, inputs):
        x = self.conv_block1(inputs)
        x = self.pool1(x)
        x = self.dropout1(x)

        x = self.conv_block2(x)
        x = self.pool2(x)
        x = self.dropout2(x)

        x = self.global_avg_pool(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal(shape=(1, 28, 28, 1))
    output_tensor = model(input_tensor)