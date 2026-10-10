import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, UpSampling2D, Concatenate, Dense, Activation, BatchNormalization, GlobalAveragePooling2D

class DL_Model(tf.keras.Model):
    def __init__(self):
        super(DL_Model, self).__init__()
        # Downsampling stages
        self.conv1 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')
        self.bn1 = BatchNormalization(axis=-1)
        self.pool1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')

        self.conv2 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')
        self.bn2 = BatchNormalization(axis=-1)
        self.pool2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')

        # Additional convolution and dropout
        self.conv3 = Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same')
        self.bn3 = BatchNormalization(axis=-1)
        self.dropout1 = Dropout(rate=0.5)

        # Upsampling stages with skip connections
        self.up1 = UpSampling2D(size=(2, 2))
        self.conv4 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')
        self.bn4 = BatchNormalization(axis=-1)

        self.up2 = UpSampling2D(size=(2, 2))
        self.conv5 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')
        self.bn5 = BatchNormalization(axis=-1)

        # Final classification layer
        self.conv6 = Conv2D(filters=10, kernel_size=(1, 1), activation='softmax', padding='same')

    def call(self, inputs):
        # Downsampling
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.pool2(x)

        # Additional convolution and dropout
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.dropout1(x)

        # Upsampling with skip connections
        x = self.up1(x)
        x = self.conv4(x)
        x = self.bn4(x)

        x = self.up2(x)
        x = self.conv5(x)
        x = self.bn5(x)

        # Final classification
        x = self.conv6(x)

        return x

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal(shape=(1, 32, 32, 3))  # CIFAR-10 input shape
    output_tensor = model(input_tensor)
    print("Model output shape:", output_tensor.shape)