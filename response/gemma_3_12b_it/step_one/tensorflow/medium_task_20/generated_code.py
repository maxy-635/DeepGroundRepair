import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPool2D, concatenate, Flatten, Dense
from tensorflow.keras.activations import softmax
from tensorflow.keras import Model

class DL_Model(Model):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Path 1: 1x1 convolution
        self.conv1_1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu', name='conv1_1')

        # Path 2: 1x1 -> 3x3 -> 3x3 convolutions
        self.conv2_1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu', name='conv2_1')
        self.conv2_2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', name='conv2_2')
        self.conv2_3 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', name='conv2_3')

        # Path 3: 1x1 -> 3x3 convolution
        self.conv3_1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu', name='conv3_1')
        self.conv3_2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', name='conv3_2')

        # Path 4: Max pooling -> 1x1 convolution
        self.pool4_1 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same', name='pool4_1')
        self.conv4_1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu', name='conv4_1')

        # Concatenate and dense layers
        self.flatten = Flatten(name='flatten')
        self.dense1 = Dense(units=128, activation='relu', name='dense1')
        self.output_layer = Dense(units=10, activation=softmax, name='output_layer')

    def call(self, inputs):
        # Path 1
        x1 = self.conv1_1(inputs)

        # Path 2
        x2 = self.conv2_1(inputs)
        x2 = self.conv2_2(x2)
        x2 = self.conv2_3(x2)

        # Path 3
        x3 = self.conv3_1(inputs)
        x3 = self.conv3_2(x3)

        # Path 4
        x4 = self.pool4_1(inputs)
        x4 = self.conv4_1(x4)

        # Concatenate
        merged = concatenate([x1, x2, x3, x4], axis=-1, name='concatenate')

        # Flatten and dense layers
        x = self.flatten(merged)
        x = self.dense1(x)
        output = self.output_layer(x)

        return output


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal((1, 32, 32, 3))  # Example input shape (batch_size, height, width, channels)
    output_tensor = model(input_tensor)
    print(output_tensor.shape)