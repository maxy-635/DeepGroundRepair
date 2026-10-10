import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPool2D, Add, Flatten, Dense, concatenate

class DL_Model(tf.keras.Model):
    def __init__(self):
        super(DL_Model, self).__init__()

        # First convolutional block
        self.conv1_1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', name='conv1_1')
        self.pool1 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid', name='pool1')

        # Second convolutional block
        self.conv2_1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', name='conv2_1')
        self.pool2 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid', name='pool2')

        # Third convolutional block
        self.conv3_1 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', name='conv3_1')

        # Separate convolutional path
        self.conv_separate = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', name='conv_separate')

        # Fully connected layers
        self.flatten = Flatten(name='flatten')
        self.dense1 = Dense(units=256, activation='relu', name='dense1')
        self.dense2 = Dense(units=10, activation='softmax', name='dense2')  # 10 classes for CIFAR-10

    def call(self, inputs):
        # First path
        x1 = self.conv1_1(inputs)
        x1 = self.pool1(x1)
        x1 = self.conv2_1(x1)
        x1 = self.pool2(x1)
        x1 = self.conv3_1(x1)

        # Second path
        x2 = self.conv_separate(inputs)

        # Add the outputs of the two paths
        added_output = Add()([x1, x2])

        # Flatten and pass through fully connected layers
        x = self.flatten(added_output)
        x = self.dense1(x)
        output = self.dense2(x)

        return output


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal((1, 32, 32, 3))  # Dummy input tensor for CIFAR-10 (batch_size, height, width, channels)
    output_tensor = model(input_tensor)
    print("Output shape:", output_tensor.shape)