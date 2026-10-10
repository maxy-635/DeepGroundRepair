import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPool2D, Conv2DTranspose, Concatenate, Dense, Flatten, UpSampling2D, BatchNormalization, Activation

class DL_Model(tf.keras.Model):
    def __init__(self):
        super(DL_Model, self).__init__()
        # Branch 1: Local feature extraction
        self.conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')
        self.conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')

        # Branch 2: Downsampling, convolution, and upsampling
        self.pool2 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')
        self.conv3 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')
        self.conv4 = Conv2DTranspose(filters=32, kernel_size=(2, 2), strides=(2, 2), padding='same', activation='relu')

        # Branch 3: Downsampling, convolution, and upsampling
        self.pool3 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')
        self.conv5 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')
        self.conv6 = Conv2DTranspose(filters=32, tf.keras.layers.Conv2DTranspose(filters=32, kernel_size=(2, 2), strides=(2, 2), padding='same', activation='relu')

        # Concatenate branches and final layers
        self.concatenate = Concatenate(axis=-1)
        self.conv7 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')
        self.flatten = Flatten()
        self.fc = Dense(units=10, activation='softmax')

    def call(self, inputs):
        # Branch 1
        branch1 = self.conv1(inputs)
        branch1 = self.conv2(branch1)

        # Branch 2
        branch2 = self.pool2(inputs)
        branch2 = self.conv3(branch2)
        branch2 = self.conv4(branch2)

        # Branch 3
        branch3 = self.pool3(inputs)
        branch3 = self.conv5(branch3)
        branch3 = self.conv6(branch3)

        # Concatenate branches
        concatenated = self.concatenate([branch1, branch2, branch3])
        concatenated = self.conv7(concatenated)

        # Final layers
        output = self.flatten(concatenated)
        output = self.fc(output)

        return output

if __name__ == "__main__":
    # Create a dummy input tensor with shape (batch_size, height, width, channels)
    input_tensor = tf.random.normal((1, 32, 32, 3))
    model = DL_Model()
    output_tensor = model(input_tensor)
    print("Output shape:", output_tensor.shape)