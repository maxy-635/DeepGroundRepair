import tensorflow as tf
from tensorflow.keras.layers import Conv2D, DepthwiseConv2D, SeparableConv2D, Dropout, Dense, Concatenate, Flatten, BatchNormalization, Activation
from tensorflow.keras.models import Model

class DL_Model(Model):
    def __init__(self):
        super(DL_Model, self).__init__()
        # Branch 1
        self.depthwise_conv1 = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', depth_multiplier=1, activation=None)
        self.pointwise_conv1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation=None)
        self.dropout1 = Dropout(rate=0.2)
        self.batch_norm1 = BatchNormalization(axis=-1)

        # Branch 2
        self.depthwise_conv2 = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', depth_multiplier=1, activation=None)
        self.pointwise_conv2 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation=None)
        self.dropout2 = Dropout(rate=0.2)
        self.batch_norm2 = BatchNormalization(axis=-1)

        # Branch 3
        self.depthwise_conv3 = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', depth_multiplier=1, activation=None)
        self.pointwise_conv3 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation=None)
        self.dropout3 = Dropout(rate=0.2)
        self.batch_norm3 = BatchNormalization(axis=-1)

        # Fully connected layers
        self.flatten = Flatten()
        self.dense1 = Dense(units=128, activation='relu')
        self.dropout4 = Dropout(rate=0.5)
        self.dense2 = Dense(units=10, activation='softmax')

    def call(self, inputs):
        # Branch 1
        x1 = self.depthwise_conv1(inputs)
        x1 = self.pointwise_conv1(x1)
        x1 = self.batch_norm1(x1)
        x1 = Activation('relu')(x1)
        x1 = self.dropout1(x1)

        # Branch 2
        x2 = self.depthwise_conv2(inputs)
        x2 = self.pointwise_conv2(x2)
        x2 = self.batch_norm2(x2)
        x2 = Activation('relu')(x2)
        x2 = self.dropout2(x2)

        # Branch 3
        x3 = self.depthwise_conv3(inputs)
        x3 = self.pointwise_conv3(x3)
        x3 = self.batch_norm3(x3)
        x3 = Activation('relu')(x3)
        x3 = self.dropout3(x3)

        # Concatenate branches
        concatenated = Concatenate(axis=-1)([x1, x2, x3])

        # Fully connected layers
        x = self.flatten(concatenated)
        x = self.dense1(x)
        x = self.dropout4(x)
        x = self.dense2(x)

        return x

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.keras.Input(shape=(28, 28, 1))  # MNIST image shape
    output_tensor = model(input_tensor)
    print("Model summary:")
    model.summary()