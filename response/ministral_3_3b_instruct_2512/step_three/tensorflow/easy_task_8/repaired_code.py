import tensorflow as tf
from tensorflow.keras.layers import SeparableConv2D, Conv1D, Conv2D, Conv2DTranspose, Dense, Dropout, Flatten, MaxPool2D, GlobalAveragePooling2D

class DL_Model(tf.keras.Model):
    def __init__(self):
        super(DL_Model, self).__init__()
        self.conv1 = SeparableConv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')
        self.conv2 = Conv1D(filters=32, kernel_size=(1, 3), strides=(1, 1), padding='same', activation='relu')
        self.dropout1 = Dropout(rate=0.2)
        self.conv3 = SeparableConv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')
        self.dropout2 = Dropout(rate=0.2)
        self.flatten = Flatten()
        self.fc = Dense(units=10, activation='softmax')

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.dropout1(x)
        x = self.conv2(x)
        x = self.dropout2(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal(shape=(1, 28, 28, 1))  # MNIST image shape: (batch_size, height, width, channels)
    output_tensor = model(input_tensor)
    print(output_tensor.shape)