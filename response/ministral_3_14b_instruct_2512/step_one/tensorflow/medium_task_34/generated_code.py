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
        self.maxpool3 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')

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
        # First part: feature extraction
        x = self.conv1(inputs)
        x = self.maxpool1(x)
        x1 = x  # Save for skip connection

        x = self.conv2(x)
        x = self.maxpool2(x)
        x2 = x  # Save for skip connection

        x = self.conv3(x)
        x = self.maxpool3(x)
        x3 = x  # Save for skip connection

        # Second part: enhance generalization
        x = self.conv4(x)
        x = self.dropout(x)
        x = self.conv5(x)

        # Third part: upsampling with skip connections
        x = self.conv6(x)
        x = self.conv6_transpose(x)
        x = Add()([x, x3])  # Skip connection

        x = self.conv7(x)
        x = self.conv7_transpose(x)
        x = Add()([x, x2])  # Skip connection

        x = self.conv8(x)
        x = self.conv8_transpose(x)
        x = Add()([x, x1])  # Skip connection

        # Final classification
        x = self.conv_final(x)

        return x

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal((1, 32, 32, 3))  # CIFAR-10 image shape: 32x32x3
    output_tensor = model(input_tensor)
    print("Output shape:", output_tensor.shape)