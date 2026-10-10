import tensorflow as tf
from tensorflow.keras.layers import Conv2D, AveragePooling2D, Flatten, Dense, Dropout, Softmax
from tensorflow.keras.models import Model

class DL_Model(Model):
    def __init__(self):
        super(DL_Model, self).__init__()
        # First two feature extraction layers
        self.conv1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')
        self.avg_pool1 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')

        self.conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')
        self.avg_pool2 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')

        # Additional three convolutional layers
        self.conv3 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 2), padding='same', activation='relu')
        self.conv4 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')
        self.conv5 = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')

        # Additional average pooling layer
        self.avg_pool3 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')

        # Fully connected layers
        self.flatten = Flatten()
        self.dense1 = Dense(units=512, activation='relu')
        self.dropout1 = Dropout(rate=0.5)
        self.dense2 = Dense(units=256, activation='relu')
        self.dropout2 = Dropout(rate=0.5)

        # Output layer
        self.softmax = Softmax(axis=-1)

    def call(self, inputs):
        # First two feature extraction layers
        x = self.conv1(inputs)
        x = self.avg_pool1(x)

        x = self.conv2(x)
        x = self.avg_pool2(x)

        # Additional three convolutional layers
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)

        # Additional average pooling layer
        x = self.avg_pool3(x)

        # Fully connected layers
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dropout1(x)
        x = self.dense2(x)
        x = self.dropout2(x)

        # Output layer
        x = self.softmax(x)
        return x

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal(shape=(1, 224, 224, 3))
    output_tensor = model(input_tensor)
    print("Input shape:", input_tensor.shape)
    print("Output shape:", output_tensor.shape)