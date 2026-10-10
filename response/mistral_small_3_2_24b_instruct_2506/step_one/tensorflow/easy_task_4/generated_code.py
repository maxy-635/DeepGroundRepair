import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.models import Model

class DL_Model(Model):
    def __init__(self):
        super(DL_Model, self).__init__()
        # First block: three convolutional layers followed by max pooling
        self.conv1 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')
        self.conv2 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')
        self.conv3 = Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same')
        self.pool1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')

        # Second block: four convolutional layers followed by max pooling
        self.conv4 = Conv2D(filters=256, kernel_size=(3, 3), activation='relu', padding='same')
        self.conv5 = Conv2D(filters=256, kernel_size=(3, 3), activation='relu', padding='same')
        self.conv6 = Conv2D(filters=512, kernel_size=(3, 3), activation='relu', padding='same')
        self.conv7 = Conv2D(filters=512, kernel_size=(3, 3), activation='relu', padding='same')
        self.pool2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')

        # Flatten and fully connected layers
        self.flatten = Flatten()
        self.fc1 = Dense(units=1024, activation='relu')
        self.fc2 = Dense(units=512, activation='relu')
        self.fc3 = Dense(units=10, activation='softmax')

    def call(self, inputs):
        # First block
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.pool1(x)

        # Second block
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.pool2(x)

        # Flatten and fully connected layers
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)

        return x

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal(shape=(1, 28, 28, 1))  # MNIST image shape: (28, 28, 1)
    output_tensor = model(input_tensor)
    print("Output shape:", output_tensor.shape)