import tensorflow as tf
from tensorflow.keras.layers import Conv2D, AveragePooling2D, Dropout, GlobalAveragePooling2D, Flatten, Dense
from tensorflow.keras.models import Model

class DL_Model(Model):
    def __init__(self):
        super(DL_Model, self).__init__()
        # First block
        self.conv3x3_1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')
        self.conv1x1_1 = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='valid', activation='relu')
        self.conv1x1_2 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='valid', activation='relu')
        self.avg_pool_1 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')
        self.dropout_1 = Dropout(rate=0.25)

        # Second block
        self.conv3x3_2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 2), padding='same', activation='relu')
        self.conv1x1_3 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='valid', activation='relu')
        self.conv1x1_4 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='valid', activation='relu')
        self.avg_pool_2 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')
        self.dropout_2 = Dropout(rate=0.25)

        # Classification head
        self.global_avg_pool = GlobalAveragePooling2D()
        self.flatten = Flatten()
        self.dense = Dense(units=10, activation='softmax')

    def call(self, inputs):
        # First block
        x = self.conv3x3_1(inputs)
        x = self.conv1x1_1(x)
        x = self.conv1x1_2(x)
        x = self.avg_pool_1(x)
        x = self.dropout_1(x)

        # Second block
        x = self.conv3x3_2(x)
        x = self.conv1x1_3(x)
        x = self.conv1x1_4(x)
        x = self.avg_pool_2(x)
        x = self.dropout_2(x)

        # Classification head
        x = self.global_avg_pool(x)
        x = self.flatten(x)
        x = self.dense(x)

        return x

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal(shape=(1, 28, 28, 1))  # MNIST image shape: 28x28 with 1 channel
    output_tensor = model(input_tensor)
    print(output_tensor.shape)  # Expected output shape: (1, 10)