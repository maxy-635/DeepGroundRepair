import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Dropout, Add, Flatten, Dense, concatenate, Reshape

class DL_Model(tf.keras.Model):
    def __init__(self):
        super(DL_Model, self).__init__()
        # Main path
        self.conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')
        self.dropout1 = Dropout(rate=0.2)
        self.conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')
        self.dropout2 = Dropout(rate=0.2)
        self.conv3 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')

        # Branch path
        self.branch_conv = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')

        # Final layers
        self.add = Add()
        self.flatten = Flatten()
        self.dense = Dense(units=10, activation='softmax')

    def call(self, inputs):
        # Main path
        x = self.conv1(inputs)
        x = self.dropout1(x)
        x = self.conv2(x)
        x = self.dropout2(x)
        x = self.conv3(x)

        # Branch path
        branch = self.branch_conv(inputs)

        # Combine paths
        combined = self.add([x, branch])

        # Final layers
        x = self.flatten(combined)
        x = self.dense(x)

        return x

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal(shape=(1, 28, 28, 1))  # MNIST images are 28x28 with 1 channel
    output_tensor = model(input_tensor)
    print(output_tensor.shape)  # Expected output shape: (1, 10)