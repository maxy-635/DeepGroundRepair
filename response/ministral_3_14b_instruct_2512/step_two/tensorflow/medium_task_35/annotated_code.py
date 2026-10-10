import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dropout, Conv2DTranspose, UpSampling2D, Concatenate, Dense, Flatten, Activation, BatchNormalization
from tensorflow.keras.models import Model

class DL_Model(Model):
    def __init__(self):
        super(DL_Model, self).__init__()
        # Downsampling path
        self.conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')
        self.conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')
        self.pool1 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')

        self.conv3 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')
        self.conv4 = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')
        self.pool2 = MaxPool2D(pool_size='mask', strides='mask', padding='mask')

        # Bottleneck
        self.conv5 = Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')
        self.dropout1 = Dropout(rate=0.5)

        # Upsampling path with skip connections
        self.up1 = UpSampling2D(size=(2, 2))
        self.conv6 = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')

        self.up2 = UpSampling2D(size=(2, 2))
        self.conv7 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')

        # Final classification layers
        self.conv8 = Conv2D(filters=10, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='softmax')

    def call(self, inputs):
        # the initial shape of inputs is: (1, 32, 32, 3)
        # Downsampling path
        x = self.conv1(inputs)  # output tensor shape: (1, 32, 32, 32)
        x = self.conv2(x)  # output tensor shape: (1, 32, 32, 64)
        x = self.pool1(x)  # output tensor shape: (1, 16, 16, 64)

        skip1 = x  # output tensor shape: (1, 16, 16, 64)
        x = self.conv3(x)  # output tensor shape: (1, 16, 16, 128)
        x = self.conv4(x)  # output tensor shape: (1, 16, 16, 256)
        x = self.pool2(x)  # output tensor shape of this layer should be (1, 16, 16, 256)

        skip2 = x
        x = self.conv5(x)  # output tensor shape: (1, 8, 8, 512)
        x = self.dropout1(x)  # output tensor shape: (1, 8, 8, 512)

        # Upsampling path with skip connections
        x = self.up1(x)  # output tensor shape: (1, 16, 16, 512)
        x = self.conv6(x)  # output tensor shape: (1, 16, 16, 256)
        x = Concatenate(axis=-1)([x, skip2])

        x = self.up2(x)
        x = self.conv7(x)
        x = Concatenate(axis=-1)([x, skip1])

        # Final classification
        x = self.conv8(x)

        return x

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal(shape=(1, 32, 32, 3))  # CIFAR-10 image shape: 32x32x3
    output_tensor = model(input_tensor)
    print("Input shape:", input_tensor.shape)
    print("Output shape:", output_tensor.shape)