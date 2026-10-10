import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Dropout, Concatenate, MaxPool2D, Flatten, Dense
from tensorflow.keras.models import Model

class DL_Model(Model):
    def __init__(self):
        super(DL_Model, self).__init__()
        # Block 1
        self.conv1x1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')
        self.conv3x3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')
        self.conv5x5 = Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')
        self.dropout = Dropout(rate=0.5)
        self.concat1 = Concatenate(axis=-1)

        # Block 2
        self.branch1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')
        self.branch2_conv1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')
        self.branch2_conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')
        self.branch3_conv1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')
        self.branch3_conv2 = Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')
        self.branch4_pool = MaxPool2D(pool_size=(3, 3), strides=(1, 1), padding='same')
        self.branch4_conv = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')
        self.concat2 = Concatenate(axis=-1)

        # Final layers
        self.flatten = Flatten()
        self.dense = Dense(units=10, activation='softmax')

    def call(self, inputs):
        # Block 1
        x1 = self.conv1x1(inputs[:, :, :, :1])
        x2 = self.conv3x3(inputs[:, :, :, 1:2])
        x3 = self.conv5x5(inputs[:, :, :, 2:3])
        x = self.concat1([x1, x2, x3])
        x = self.dropout(x)

        # Block 2
        branch1 = self.branch1(x)
        branch2 = self.branch2_conv2(self.branch2_conv1(x))
        branch3 = self.branch3_conv2(self.branch3_conv1(x))
        branch4 = self.branch4_conv(self.branch4_pool(x))
        x = self.concat2([branch1, branch2, branch3, branch4])

        # Final layers
        x = self.flatten(x)
        return self.dense(x)

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal(shape=(1, 32, 32, 3))
    output_tensor = model(input_tensor)
    print("Input shape:", input_tensor.shape)
    print("Output shape:", output_tensor.shape)