import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dropout, Conv2DTranspose, Dense, Add, GlobalAveragePooling2D, Activation, BatchNormalization, Flatten

class DL_Model(tf.keras.Model):
    def __init__(self):
        super(DL_Model, self).__init__()

        # First part: 3 pairs of Conv2D + MaxPool2D
        self.conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')
        self.maxpool1 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')

        self.conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')
        self.maxpool2 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')

        self.conv3 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')
        self.maxpool3 = MaxPool2D(pool_size='mask', strides='mask', padding='mask')

        # Second part: Conv2D + Dropout + Conv2D
        self.conv4 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')
        self.dropout = Dropout(rate=0.5)
        self.conv5 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')

        # Third part: 3 pairs of Conv2D + Conv2DTranspose with skip connections
        self.conv6 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')
        self.conv6_transpose = Conv2DTranspose(filters=128, kernel_size=(3, 3), strides=(2, 2), padding='same', activation='relu')

        self.conv7 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')
        self.conv7_transpose = Conv2DTranspose(filters=64, kernel_size=(3, 3), strides=(2, 2), padding='same', activation='relu')

        self.conv8 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')
        self.conv8_transpose = Conv2DTranspose(filters=32, kernel_size=(3, 3), strides=(2, 2), padding='same', activation='relu')

        # Final 1x1 convolution for classification
        self.conv_final = Conv2D(filters=10, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='softmax')

    def call(self, inputs):
        # the initial shape of inputs is: (1, 32, 32, 3)
        # First part: feature extraction
        x = self.conv1(inputs)  # output tensor shape: (1, 32, 32, 32)
        x = self.maxpool1(x)  # output tensor shape: (1, 16, 16, 32)
        x1 = x  # output tensor shape: (1, 16, 16, 32)

        x = self.conv2(x)  # output tensor shape: (1, 16, 16, 64)
        x = self.maxpool2(x)  # output tensor shape: (1, 8, 8, 64)
        x2 = x  # output tensor shape: (1, 8, 8, 64)

        x = self.conv3(x)  # output tensor shape: (1, 8, 8, 128)
        x = self.maxpool3(x)  # output tensor shape of this layer should be (1, 8, 8, 128)
        x3 = x

        # Second part: enhance generalization
        x = self.conv4(x)  # output tensor shape: (1, 4, 4, 128)
        x = self.dropout(x)  # output tensor shape: (1, 4, 4, 128)
        x = self.conv5(x)  # output tensor shape: (1, 4, 4, 128)

        # Third part: upsampling with skip connections
        x = self.conv6(x)  # output tensor shape: (1, 4, 4, 128)
        x = self.conv6_transpose(x)  # output tensor shape: (1, 8, 8, 128)
        x = Add()([x, x3])

        x = self.conv7(x)
        x = self.conv7_transpose(x)
        x = Add()([x, x2])

        x = self.conv8(x)
        x = self.conv8_transpose(x)
        x = Add()([x, x1])

        # Final classification
        x = self.conv_final(x)

        return x

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal((1, 32, 32, 3))  # CIFAR-10 image shape: 32x32x3
    output_tensor = model(input_tensor)
    print("Output shape:", output_tensor.shape)