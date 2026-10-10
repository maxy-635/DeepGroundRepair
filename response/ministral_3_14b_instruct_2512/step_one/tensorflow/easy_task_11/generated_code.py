import tensorflow as tf
from tensorflow.keras.layers import AveragePooling2D, Conv2D, Flatten, Dense, Dropout
from tensorflow.keras.models import Model

class DL_Model(Model):
    def __init__(self):
        super(DL_Model, self).__init__()
        self.average_pooling = AveragePooling2D(pool_size=(5, 5), strides=(3, 3), padding='valid')
        self.conv1x1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='valid', activation='relu')
        self.flatten = Flatten()
        self.dense1 = Dense(units=128, activation='relu')
        self.dropout = Dropout(rate=0.5)
        self.dense2 = Dense(units=10, activation='softmax')

    def call(self, inputs):
        x = self.average_pooling(inputs)
        x = self.conv1x1(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dropout(x)
        x = self.dense2(x)
        return x

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal(shape=(32, 28, 28, 1))  # MNIST images are 28x28 with 1 channel
    output_tensor = model(input_tensor)
    print(output_tensor.shape)