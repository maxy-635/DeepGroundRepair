import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dropout, Add, GlobalAveragePooling2D, Flatten, Dense

class DL_Model(tf.keras.Model):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Main pathway
        self.conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')
        self.conv2_1 = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')
        self.conv2_2 = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')
        self.maxpool = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')
        self.dropout = Dropout(rate=0.5)

        # Branch pathway
        self.conv_branch = Conv2D(filters=16, kernel_size=(3, 3), strides=(2, 2), padding='same', activation='relu')

        # Fusion and final layers
        self.add = Add()
        self.gap = GlobalAveragePooling2D(keepdims=False)
        self.flatten = Flatten()
        self.dense = Dense(units=10, activation='softmax')

    def call(self, inputs):
        # the initial shape of inputs is: (1, 28, 28, 1)
        # Main pathway
        x = self.conv1(inputs)  # output tensor shape: (1, 28, 28, 32)
        x = self.conv2_1(x)  # output tensor shape: (1, 28, 28, 16)
        x = self.conv2_2(x)  # output tensor shape: (1, 28, 28, 16)
        x = self.maxpool(x)  # output tensor shape: (1, 14, 14, 16)
        x = self.dropout(x)  # output tensor shape: (1, 14, 14, 16)

        # Branch pathway
        branch = self.conv_branch(inputs)  # output tensor shape of this layer should be (1, 14, 14, 16)

        # Fusion
        merged = self.add([x, branch])

        # Final layers
        x = self.gap(merged)
        x = self.flatten(x)
        output = self.dense(x)

        return output


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal((1, 28, 28, 1))  # Dummy input tensor (batch_size, height, width, channels)
    output_tensor = model(input_tensor)
    print(output_tensor.shape)