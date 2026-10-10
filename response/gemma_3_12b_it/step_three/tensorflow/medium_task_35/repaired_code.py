import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dropout, Conv2DTranspose, concatenate, Activation, BatchNormalization, Reshape, Flatten, Dense

class DL_Model(tf.keras.Model):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Stage 1: Convolution and Max Pooling
        self.conv1_1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', name='conv1_1')
        self.pool1 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same', name='pool1')

        self.conv1_2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', name='conv1_2')
        self.pool2 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same', name='pool2')

        # Stage 2: Convolution and Dropout
        self.conv2_1 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', name='conv2_1')
        self.dropout1 = Dropout(rate=0.25, name='dropout1')

        self.conv2_2 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', name='conv2_2')
        self.dropout2 = Dropout(rate=0.25, name='dropout2')

        # Stage 3: Upsampling with Skip Connections
        self.conv_up1 = Conv2DTranspose(filters=64, kernel_size=(3, 3), strides=(2, 2), padding='same', activation='relu', name='conv_up1')
        self.concatenate1 = concatenate(inputs=x, axis=-1, name='mask')

        self.conv_up2 = Conv2DTranspose(filters=64, kernel_size=(3, 3), strides=(2, 2), padding='same', activation='relu', name='conv_up2')
        self.concatenate2 = concatenate([self.conv1_1.output, self.conv_up2.output], axis=-1, name='concatenate2')

        # Output Layer
        self.conv_out = Conv2D(filters=10, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='linear', name='conv_out')


    def call(self, inputs):
        x = self.conv1_1(inputs)
        x = self.pool1(x)
        x = self.conv1_2(x)
        x = self.pool2(x)

        x = self.conv2_1(x)
        x = self.dropout1(x)
        x = self.conv2_2(x)
        x = self.dropout2(x)

        x = self.conv_up1(x)
        x = self.concatenate1(x)
        x = self.conv_up2(x)
        x = self.concatenate2(x)

        x = self.conv_out(x)
        return x


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal((1, 32, 32, 3))  # Example input shape (batch_size, height, width, channels)
    output_tensor = model(input_tensor)
    print("Output shape:", output_tensor.shape)