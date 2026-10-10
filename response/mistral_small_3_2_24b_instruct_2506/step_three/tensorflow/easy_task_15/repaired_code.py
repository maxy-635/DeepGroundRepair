import tensorflow as tf
from tensorflow.keras.layers import Conv2D, AveragePooling2D, Dropout, GlobalAveragePooling2D, Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras import Input

class DL_Model(Model):
    def __init__(self):
        super(DL_Model, self).__init__()
        # First specialized block
        self.conv1 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')
        self.conv2 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu', padding='same')
        self.conv3 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu', padding='same')
        self.avg_pool1 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')
        self.dropout1 = Dropout(rate=0.5)

        # Second specialized block
        self.conv4 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')
        self.conv5 = Conv2D(filters=64, kernel_size=(1, 1), activation='relu', padding='same')
        self.conv6 = Conv2D(filters=64, kernel_size=(1, 1), activation='relu', padding='same')
        self.avg_pool2 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')
        self.dropout2 = Dropout(rate=0.5)

        # Global average pooling and dense layer
        self.global_avg_pool = GlobalAveragePooling2D(keepdims=False)
        self.flatten = Flatten()
        self.dense = Dense(units=10, activation='softmax')

    def call(self, inputs):
        # First block
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.avg_pool1(x)
        x = self.dropout1(x)

        # Second block
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.avg_pool2(x)
        x = self.dropout2(x)

        # Global average pooling and dense layer
        x = self.global_avg_pool(x)
        x = self.flatten(x)
        x = self.dense(x)
        return x

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal(shape=(1, 28, 28, 1))
    output_tensor = model(input_tensor)
    print(output_tensor)