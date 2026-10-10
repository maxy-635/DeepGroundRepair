import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input

class DL_Model(Model):
    def __init__(self):
        super(DL_Model, self).__init__()
        # Main path
        self.conv1 = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu')
        self.conv2 = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu')
        self.max_pool = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid')

        # Branch path
        self.branch_conv = tf.keras.layers.Conv2D(filters=32, kernel_size=(5, 5), activation='relu')
        self.branch_pool = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid')

        # Fully connected layers
        self.flatten = Flatten()
        self.fc1 = Dense(units=128, activation='relu')
        self.fc2 = Dense(units=10, activation='softmax')

    def call(self, inputs):
        # Main path
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.max_pool(x)

        # Branch path
        y = self.branch_conv(inputs)
        y = self.branch_pool(y)

        # Concatenate features from both paths
        combined = tf.concat([x, y], axis=-1)

        # Fully connected layers
        z = self.flatten(combined)
        z = self.fc1(z)
        output = self.fc2(z)

        return output

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.keras.Input(shape=(32, 32, 3))
    output_tensor = model(input_tensor)
    print(output_tensor)