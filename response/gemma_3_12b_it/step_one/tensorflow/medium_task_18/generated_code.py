import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPool2D, Concatenate, Flatten, Dense

class DL_Model(tf.keras.Model):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Feature extraction layers
        self.conv1_1x1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu', name='conv1_1x1')
        self.pool1 = MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='same', name='pool1')

        self.conv2_3x3 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', name='conv2_3x3')
        self.pool2 = MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='same', name='pool2')

        self.conv3_5x5 = Conv2D(filters=256, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu', name='conv3_5x5')
        self.pool3 = MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='same', name='pool3')

        # Feature combination
        self.concat = Concatenate(axis=-1, name='concat')

        # Classification layers
        self.flatten = Flatten(name='flatten')
        self.dense1 = Dense(units=512, activation='relu', name='dense1')
        self.dense2 = Dense(units=10, activation='softmax', name='dense2')  # 10 classes for CIFAR-10

    def call(self, inputs):
        x = self.conv1_1x1(inputs)
        x = self.pool1(x)

        x = self.conv2_3x3(x)
        x = self.pool2(x)

        x = self.conv3_5x5(x)
        x = self.pool3(x)

        x = self.concat([x])  # Concatenate the features from different scales

        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dense2(x)

        return x


if __name__ == "__main__":
    model = DL_Model()
    # CIFAR-10 images are 32x32 with 3 channels (RGB)
    input_tensor = tf.random.normal(shape=(1, 32, 32, 3))  # Batch size of 1
    output_tensor = model(input_tensor)
    print(output_tensor.shape)