import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dropout, Conv2DTranspose, Reshape, Flatten, Dense, Activation, BatchNormalization

class DL_Model(tf.keras.Model):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Feature Extraction Block 1
        self.conv1_1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='he_normal')
        self.pool1 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')

        self.conv2_1 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='he_normal')
        self.pool2 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')

        self.conv3_1 = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='he_normal')
        self.pool3 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')

        # Generalization Enhancement Block
        self.conv4_1 = Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='he_normal')
        self.dropout = Dropout(rate=0.25)
        self.conv4_2 = Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='he_normal')

        # Upsampling and Skip Connection Block
        self.conv5_1 = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='he_normal')
        self.conv5_2 = Conv2DTranspose(filters=256, kernel_size=(3, 3), strides=(2, 2), padding='same', activation='relu', kernel_initializer='he_normal')

        self.conv6_1 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='he_normal')
        self.conv6_2 = Conv2DTranspose(filters=128, kernel_size=(3, 3), strides=(2, 2), padding='same', activation='relu', kernel_initializer='he_normal')

        self.conv7_1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='he_normal')
        self.conv7_2 = Conv2DTranspose(filters=64, kernel_size=(3, 3), strides=(2, 2), padding='same', activation='relu', kernel_initializer='he_normal')

        # Output Layer
        self.conv8_1 = Conv2D(filters=10, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='linear', kernel_initializer='he_normal')

    def call(self, inputs):
        # the initial shape of inputs is: (1, 32, 32, 3)
        # Feature Extraction
        x = self.conv1_1(inputs)  # output tensor shape: (1, 32, 32, 64)
        x = self.pool1(x)  # output tensor shape: (1, 16, 16, 64)
        x = self.conv2_1(x)  # output tensor shape: (1, 16, 16, 128)
        x = self.pool2(x)  # output tensor shape: (1, 8, 8, 128)
        x = self.conv3_1(x)  # output tensor shape: (1, 8, 8, 256)
        x = self.pool3(x)  # output tensor shape: (1, 4, 4, 256)

        # Generalization Enhancement
        x = self.conv4_1(x)  # output tensor shape: (1, 4, 4, 512)
        x = self.dropout(x)  # output tensor shape: (1, 4, 4, 512)
        x = self.conv4_2(x)  # output tensor shape: (1, 4, 4, 512)

        # Upsampling with Skip Connections
        x = self.conv5_1(x)  # output tensor shape: (1, 4, 4, 256)
        x = self.conv5_2(x)  # output tensor shape: (1, 8, 8, 256)
        x = tf.concat([x, self.conv3_1.get_weights()[0]], axis=-1)

        x = self.conv6_1(x)
        x = self.conv6_2(x)
        x = tf.concat([x, self.conv2_1.get_weights()[0]], axis=-1)

        x = self.conv7_1(x)
        x = self.conv7_2(x)
        x = tf.concat([x, self.conv1_1.get_weights()[0]], axis=-1)

        # Output Layer
        x = self.conv8_1(x)
        return x


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal((1, 32, 32, 3))  # Example input shape (batch_size, height, width, channels)
    output_tensor = model(input_tensor)
    print(output_tensor.shape)