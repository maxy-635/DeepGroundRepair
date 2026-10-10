import tensorflow as tf
from tensorflow.keras.layers import Layer, Conv2D, Dropout, Flatten, Dense
from tensorflow.keras.models import Model

class DL_Model(Model):
    def __init__(self):
        super(DL_Model, self).__init__()
        # Two 1x1 convolutional layers
        self.conv1x1_1 = Conv2D(filters=64, kernel_size=(1, 1), padding='same', activation='relu')
        self.dropout1 = Dropout(rate=0.2)

        self.conv1x1_2 = Conv2D(filters=64, kernel_size=(1, 1), padding='same', activation='relu')
        self.dropout2 = Dropout(rate=0.2)

        # 3x1 and 1x3 convolutional layers
        self.conv3x1 = Conv2D(filters=64, kernel_size=(3, 1), padding='same', activation='relu')
        self.dropout3 = Dropout(rate=0.2)

        self.conv1x3 = Conv2D(filters=64, kernel_size=(1, 3), padding='same', activation='relu')
        self.dropout4 = Dropout(rate=0.2)

        # Restore channels to match input
        self.conv1x1_3 = Conv2D(filters=1, kernel_size=(1, 1), padding='same', activation='relu')
        self.dropout5 = Dropout(rate=0.2)

        # Flatten and fully connected layer
        self.flatten = Flatten()
        self.dense = Dense(units=10, activation='softmax')

    def call(self, inputs):
        # Store original input
        original_input = inputs

        # First 1x1 convolution
        x = self.conv1x1_1(inputs)
        x = self.dropout1(x)

        # Second 1x1 convolution
        x = self.conv1x1_2(x)
        x = self.dropout2(x)

        # 3x1 convolution
        x = self.conv3x1(x)
        x = self.dropout3(x)

        # 1x3 convolution
        x = self.conv1x3(x)
        x = self.dropout4(x)

        # Restore channels
        x = self.conv1x1_3(x)
        x = self.dropout5(x)

        # Combine with original input
        x = tf.math.add(x, original_input)

        # Flatten and dense layer
        x = self.flatten(x)
        x = self.dense(x)

        return x

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal(shape=(1, 28, 28, 1))  # MNIST image shape
    output_tensor = model(input_tensor)
    print("Input shape:", input_tensor.shape)
    print("Output shape:", output_tensor.shape)