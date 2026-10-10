import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Conv2DTranspose

class DL_Model(tf.keras.Model):
    def __init__(self):
        super(DL_Model, self).__init__()
        # First 1x1 convolution to reduce dimensionality
        self.conv1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='valid', activation='relu')
        # 3x3 convolution to extract features
        self.conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')
        # Another 1x1 convolution to restore dimensionality
        self.conv3 = Conv2DTranspose(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='valid', activation='relu')
        # Flatten the output
        self.flatten = tf.keras.layers.Flatten()
        # Fully connected layer for classification
        self.fc = tf.keras.layers.Dense(units=10, activation='softmax')

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x

if __name__ == "__main__":
    # Create a dummy input tensor with shape (batch_size, height, width, channels)
    input_tensor = tf.random.normal(shape=(1, 28, 28, 1))
    model = DL_Model()
    output_tensor = model(input_tensor)
    print("Input shape:", input_tensor.shape)
    print("Output shape:", output_tensor.shape)