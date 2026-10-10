import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dropout, UpSampling2D

class DL_Model(tf.keras.Model):
    def __init__(self):
        super(DL_Model, self).__init__()
        # Encoder part: 3 pairs of <Conv2D, MaxPool2D>
        self.conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')
        self.pool1 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')

        self.conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')
        self.pool2 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')

        self.conv3 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')
        self.pool3 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')

        # Middle part: <Conv2D, Dropout, Conv2D>
        self.conv_mid1 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')
        self.dropout = Dropout(rate=0.5)
        self.conv_mid2 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')

        # Decoder part: 3 pairs of <Conv2D, UpSampling2D> with skip connections
        self.upconv1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')
        self.upconv1_transpose = tf.keras.layers.UpSampling2D(size=(2, 2))

        self.upconv2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')
        self.upconv2_transpose = tf.keras.layers.UpSampling2D(size=(2, 2))

        self.upconv3 = Conv2D(filters=10, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')

        # Skip connection layers
        self.skip_conv1 = Conv2D(filters=128, kernel_size=(1, 1), strides=(1, 1), padding='same')
        self.skip_conv2 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same')
        self.skip_conv3 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same')

    def call(self, inputs):
        # Encoder
        x1 = self.conv1(inputs)
        x1 = self.pool1(x1)

        x2 = self.conv2(x1)
        x2 = self.pool2(x2)

        x3 = self.conv3(x2)
        x3 = self.pool3(x3)

        # Middle part
        x_mid = self.conv_mid1(x3)
        x_mid = self.dropout(x_mid)
        x_mid = self.conv_mid2(x_mid)

        # Decoder with skip connections
        # First upsampling block
        x = self.upconv1(x_mid)
        x = tf.keras.layers.Concatenate()([x, self.skip_conv1(x3)])
        x = self.upconv1_transpose(x)

        # Second upsampling block
        x = self.upconv2(x)
        x = tf.keras.layers.Concatenate()([x, self.skip_conv2(x2)])
        x = self.upconv2_transpose(x)

        # Third upsampling block
        x = self.upconv3(x)
        x = tf.keras.layers.Concatenate()([x, self.skip_conv3(x1)])

        return x

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.keras.Input(shape=(32, 32, 3))
    output_tensor = model(input_tensor)
    print("Input shape:", input_tensor.shape)
    print("Output shape:", output_tensor.shape)