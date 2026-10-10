import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Dropout, Concatenate, MaxPool2D, Flatten, Dense

class DL_Model(tf.keras.Model):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Block 1
        self.conv1_1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')
        self.conv1_3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')
        self.conv1_5 = Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')
        self.dropout1 = Dropout(rate=0.25)
        self.concat1 = Concatenate(axis=-1)

        # Block 2
        self.conv2_1_branch1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')
        self.conv2_1_branch2 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')
        self.conv2_3_branch1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')
        self.conv2_5_branch1 = Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')
        self.maxpool = MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='valid')
        self.conv2_1_branch3 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')
        self.concat2 = Concatenate(axis=-1)

        # Output
        self.flatten = Flatten()
        self.dense = Dense(units=10, activation='softmax')

    def call(self, inputs):
        # the initial shape of inputs is: (1, 32, 32, 3)
        # Block 1
        x1 = self.conv1_1(inputs)  # output tensor shape: (1, 32, 32, 64)
        x2 = self.conv1_3(inputs)  # output tensor shape: (1, 32, 32, 64)
        x3 = self.conv1_5(inputs)  # output tensor shape: (1, 32, 32, 64)
        x = self.dropout1(Concatenate(axis=-1)([x1, x2, x3]))  # output tensor shape: (1, 32, 32, 192)

        # Block 2
        x1_branch1 = self.conv2_1_branch1(x)  # output tensor shape: (1, 32, 32, 64)
        x1_branch2 = self.conv2_1_branch2(x)  # output tensor shape: (1, 32, 32, 64)
        x1_branch3 = self.conv2_3_branch1(x)  # output tensor shape: (1, 32, 32, 64)
        x1_branch4 = self.conv2_5_branch1(x)  # output tensor shape: (1, 32, 32, 64)
        x2 = self.maxpool(x)  # output tensor shape: (1, 15, 15, 192)
        x2 = self.conv2_1_branch3(x2)  # output tensor shape: (1, 15, 15, 64)
        x = self.concat2(Concatenate(axis=-1)([x1_branch1, x1_branch2, x1_branch3, x1_branch4]))

        # Output
        x = self.flatten(x)
        x = self.dense(x)
        return x


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal((1, 32, 32, 3))  # Example input tensor (batch_size, height, width, channels)
    output_tensor = model(input_tensor)
    print(output_tensor.shape)