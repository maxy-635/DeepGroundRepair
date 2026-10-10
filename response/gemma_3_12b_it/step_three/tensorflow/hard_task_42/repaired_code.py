import tensorflow as tf
from tensorflow.keras.layers import MaxPool2D, Flatten, Dropout, Dense, Reshape, Conv2D, AveragePooling2D, Concatenate
from tensorflow.keras import Model

class DL_Model(Model):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Block 1
        self.maxpool1_1 = MaxPool2D(pool_size=(1, 1), strides=(1, 1), padding='valid')
        self.maxpool1_2 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid')
        self.maxpool1_4 = MaxPool2D(pool_size=(4, 4), strides=(4, 4), padding='valid')
        self.flatten1 = Flatten()
        self.dropout1 = Dropout(rate=0.2)

        # Concatenate outputs from parallel paths in Block 1
        self.concat1 = Concatenate()

        # Transformation layer before Block 2
        self.dense1 = Dense(units=64, activation='relu')
        self.reshape1 = Reshape(target_shape=(64,))

        # Block 2
        self.conv2_1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='valid', activation='relu')
        self.conv2_2 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='valid', activation='relu')
        self.conv2_3 = Conv2D(filters=32, kernel_size=(1, 7), strides=(1, 1), padding='valid', activation='relu')
        self.conv2_4 = Conv2D(filters=32, kernel_size=(7, 1), strides=(1, 1), padding='valid', activation='relu')
        self.conv2_5 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='valid', activation='relu')
        self.conv2_6 = Conv2D(filters=32, kernel_size=(7, 1), strides=(1, 1), padding='valid', activation='relu')
        self.conv2_7 = Conv2D(filters=32, kernel_size=(1, 7), strides=(1, 1), padding='valid', activation='relu')
        self.avgpool2_1 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')
        self.concat2 = Concatenate()

        # Classification layers
        self.dense3 = Dense(units=128, activation='relu')
        self.dense4 = Dense(units=10, activation='softmax') # Output 10 classes for MNIST

    def call(self, inputs):
        # the initial shape of inputs is: (1, 28, 28, 1)
        # Block 1
        x1_1 = self.maxpool1_1(inputs)  # output tensor shape: (1, 28, 28, 1)
        x1_2 = self.maxpool1_2(inputs)  # output tensor shape: (1, 14, 14, 1)
        x1_4 = self.maxpool1_4(inputs)  # output tensor shape: (1, 7, 7, 1)

        x1_1 = self.flatten1(x1_1)  # output tensor shape: (1, 784)
        x1_2 = self.flatten1(x1_2)  # output tensor shape: (1, 196)
        x1_4 = self.flatten1(x1_4)  # output tensor shape: (1, 49)

        x1_1 = self.dropout1(x1_1)  # output tensor shape: (1, 784)
        x1_2 = self.dropout1(x1_2)  # output tensor shape: (1, 196)
        x1_4 = self.dropout1(x1_4)  # output tensor shape: (1, 49)

        x1 = self.concat1([x1_1, x1_2, x1_4])  # output tensor shape: (1, 1029)

        # Transformation layer
        x2 = self.dense1(x1)  # output tensor shape: (1, 64)
        x2 = self.reshape1(x2)

        # Block 2
        x2_1 = self.conv2_1(x2)
        x2_2 = self.conv2_2(x2)
        x2_3 = self.conv2_3(x2)
        x2_4 = self.conv2_4(x2)
        x2_5 = self.conv2_5(x2)
        x2_6 = self.conv2_6(x2)
        x2_7 = self.conv2_7(x2)
        x2_8 = self.avgpool2_1(x2)

        x2 = self.concat2([x2_1, x2_2, x2_3, x2_4, x2_5, x2_6, x2_7, x2_8])

        # Classification layers
        x3 = self.dense3(x2)
        output = self.dense4(x3)

        return output


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal((1, 28, 28, 1))  # Example input: batch size 1, 28x28 grayscale image
    output_tensor = model(input_tensor)
    print("Output shape:", output_tensor.shape)