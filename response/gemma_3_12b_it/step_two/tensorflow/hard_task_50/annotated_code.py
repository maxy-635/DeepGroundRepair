import tensorflow as tf
from tensorflow.keras.layers import MaxPool2D, Flatten, Dropout, Concatenate, Dense, Reshape, SeparableConv2D
from tensorflow.keras.models import Model

class DL_Model(Model):
    def __init__(self):
        super(DL_Model, self).__init__()

        # First Block
        self.maxpool1 = MaxPool2D(pool_size=(1, 1), strides=(1, 1), padding='valid')
        self.maxpool2 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid')
        self.maxpool3 = MaxPool2D(pool_size=(4, 4), strides=(4, 4), padding='valid')
        self.flatten1 = Flatten()
        self.flatten2 = Flatten()
        self.flatten3 = Flatten()
        self.dropout = Dropout(rate=0.25)
        self.concat = Concatenate(axis=-1)
        self.dense1 = Dense(128, activation='relu')
        self.reshape = Reshape(target_shape='mask')

        # Second Block
        self.separable_conv1 = SeparableConv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')
        self.separable_conv2 = SeparableConv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')
        self.separable_conv3 = SeparableConv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')
        self.separable_conv4 = SeparableConv2D(filters=64, kernel_size=(7, 7), strides=(1, 1), padding='same', activation='relu')
        self.concat2 = Concatenate(axis=-1)
        self.flatten4 = Flatten()
        self.dense2 = Dense(10, activation='softmax')  # Output layer for 10 classes

    def call(self, inputs):
        # the initial shape of inputs is: (1, 32, 32, 3)
        # First Block
        x = self.maxpool1(inputs)  # output tensor shape: (1, 32, 32, 3)
        x = self.flatten1(x)  # output tensor shape: (1, 3072)
        x = self.maxpool2(inputs)  # output tensor shape: (1, 16, 16, 3)
        x = self.flatten2(x)  # output tensor shape: (1, 768)
        x = self.maxpool3(inputs)  # output tensor shape: (1, 8, 8, 3)
        x = self.flatten3(x)  # output tensor shape: (1, 192)

        x = self.dropout(x)  # output tensor shape: (1, 192)
        x = self.concat([x])  # output tensor shape: (1, 192)
        x = self.dense1(x)  # output tensor shape: (1, 128)
        x = self.reshape(x)

        # Second Block
        x = self.separable_conv1(x)
        x = self.separable_conv2(x)
        x = self.separable_conv3(x)
        x = self.separable_conv4(x)
        x = self.concat2([x])
        x = self.flatten4(x)
        x = self.dense2(x)

        return x


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal((1, 32, 32, 3))  # Dummy input tensor with shape (batch_size, height, width, channels)
    output_tensor = model(input_tensor)
    print("Output shape:", output_tensor.shape)