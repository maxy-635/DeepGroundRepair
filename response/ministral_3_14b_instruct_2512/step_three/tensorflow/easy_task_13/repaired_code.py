import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Dropout, Add, Flatten, Dense
from tensorflow.keras.models import Model

class DL_Model(Model):
    def __init__(self):
        super(DL_Model, self).__init__()
        # First 1x1 convolutional layer
        self.conv1x1_1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')
        self.dropout1 = Dropout(rate=0.2)

        # Second 1x1 convolutional layer
        self.conv1x1_2 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')
        self.dropout2 = Dropout(rate=0.2)

        # 3x1 convolutional layer
        self.conv3x1 = Conv2D(filters=64, kernel_size=(3, 1), strides=(1, 1), padding='same', activation='relu')
        self.dropout3 = Dropout(rate=0.2)

        # 1x3 convolutional layer
        self.conv1x3 = Conv2D(filters=64, kernel_size=(1, 3), strides=(1, 1), padding='same', activation='relu')
        self.dropout4 = Dropout(rate=0.2)

        # Restore channels to match input
        self.conv1x1_restore = Conv2D(filters=1, kernel_size=(1, 1), strides=(1, 1), padding='same', activation=None)

        # Add layer to combine features with original input
        self.add = Add()

        # Flatten and dense layers
        self.flatten = Flatten()
        self.dense = Dense(units=10, activation='softmax')

    def call(self, inputs):
        # First 1x1 convolutional layer with dropout
        x = self.conv1x1_1(inputs)
        x = self.dropout1(x)

        # Second 1x1 convolutional layer with dropout
        x = self.conv1x1_2(x)
        x = self.dropout2(x)

        # 3x1 convolutional layer with dropout
        x = self.conv3x1(x)
        x = self.dropout3(x)

        # 1x3 convolutional layer with dropout
        x = self.conv1x3(x)
        x = self.dropout4(x)

        # Restore channels to match input
        x = self.conv1x1_restore(x)

        # Add original input to processed features
        x = self.add([inputs, x])

        # Flatten and dense layers
        x = self.flatten(x)
        x = self.dense(x)

        return x

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal(shape=(1, 28, 28, 1))  # MNIST images are 28x28 with 1 channel
    output_tensor = model(input_tensor)
    print("Output shape:", output_tensor.shape)