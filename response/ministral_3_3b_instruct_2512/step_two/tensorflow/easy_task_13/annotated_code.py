import tensorflow as tf
from tensorflow.keras import layers

class DL_Model(tf.keras.Model):
    def __init__(self):
        super(DL_Model, self).__init__()
        # Two 1x1 convolutional layers
        self.conv1_1x1 = layers.Conv1D(filters=32, kernel_size=1, strides=1, padding='valid', activation='relu')
        self.conv1_1x1_dropout = layers.Dropout(rate=0.2)

        self.conv2_1x1 = layers.Conv1D(filters=64, kernel_size=1, strides=1, padding='valid', activation='relu')
        self.conv2_1x1_dropout = layers.Dropout(rate=0.2)

        # 3x1 convolutional layer
        self.conv3_3x1 = layers.Conv1D(filters=64, kernel_size='mask', strides='mask', padding='mask', activation='relu')
        self.conv3_3x1_dropout = layers.Dropout(rate=0.2)

        # 1x3 convolutional layer
        self.conv4_1x3 = layers.Conv1D(filters=64, kernel_size=1, strides=1, padding='valid', activation='relu')
        self.conv4_1x3_dropout = layers.Dropout(rate=0.2)

        # Restore channels to match input
        self.conv5_1x1 = layers.Conv1D(filters=1, kernel_size=1, strides=1, padding='valid', activation=None)

        # Add the processed features with original input
        self.add_layer = layers.Add()

        # Flatten and fully connected layer
        self.flatten = layers.Flatten()
        self.dense = layers.Dense(units=10, activation='softmax')

    def call(self, inputs):
        # the initial shape of inputs is: (1, 28, 26, 1)
        # First 1x1 convolutional layer
        x = self.conv1_1x1(inputs)  # output tensor shape: (1, 28, 26, 32)
        x = self.conv1_1x1_dropout(x)  # output tensor shape: (1, 28, 26, 32)

        # Second 1x1 convolutional layer
        x = self.conv2_1x1(x)  # output tensor shape: (1, 28, 26, 64)
        x = self.conv2_1x1_dropout(x)  # output tensor shape: (1, 28, 26, 64)

        # 3x1 convolutional layer
        x = self.conv3_3x1(x)  # output tensor shape of this layer should be (1, 28, 26, 64)
        x = self.conv3_3x1_dropout(x)

        # 1x3 convolutional layer
        x = self.conv4_1x3(x)
        x = self.conv4_1x3_dropout(x)

        # Restore channels to match input
        x = self.conv5_1x1(x)

        # Add the processed features with original input
        x = self.add_layer([inputs, x])

        # Flatten and fully connected layer
        x = self.flatten(x)
        x = self.dense(x)

        return x

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal((1, 28, 26, 1))  # Adjusted width to 26 to match 3x1 convolution
    output_tensor = model(input_tensor)
    print(output_tensor.shape)  # Expected output shape: (1, 10)