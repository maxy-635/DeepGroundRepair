import tensorflow as tf
from tensorflow.keras.layers import Conv2D, AveragePooling2D, Flatten, Dense, Add, concatenate, Reshape

class DL_Model(tf.keras.Model):
    def __init__(self):
        super(DL_Model, self).__init__()
        # Path1: Two blocks of convolution followed by average pooling
        self.conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')
        self.avg_pool1 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')
        self.conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')
        self.avg_pool2 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')

        # Path2: Single convolutional layer
        self.conv3 = Conv2D(filters='mask', kernel_size='mask', strides='mask', padding='mask', activation='relu')

        # Fully connected layer
        self.flatten = Flatten()
        self.dense = Dense(units=10, activation='softmax')

    def call(self, inputs):
        # the initial shape of inputs is: (1, 32, 32, 3)
        # Path1 processing
        x1 = self.conv1(inputs)  # output tensor shape: (1, 32, 32, 32)
        x1 = self.avg_pool1(x1)  # output tensor shape: (1, 16, 16, 32)
        x1 = self.conv2(x1)  # output tensor shape: (1, 16, 16, 64)
        x1 = self.avg_pool2(x1)  # output tensor shape: (1, 8, 8, 64)

        # Path2 processing
        x2 = self.conv3(inputs)  # output tensor shape of this layer should be (1, 8, 8, 64)

        # Combine the outputs from both paths
        combined = Add()([x1, x2])

        # Flatten and pass through dense layer
        flattened = self.flatten(combined)
        output = self.dense(flattened)

        return output

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal(shape=(1, 32, 32, 3))  # Dummy input tensor for CIFAR-10 (batch_size=1, height=32, width=32, channels=3)
    output_tensor = model(input_tensor)
    print("Output tensor shape:", output_tensor.shape)