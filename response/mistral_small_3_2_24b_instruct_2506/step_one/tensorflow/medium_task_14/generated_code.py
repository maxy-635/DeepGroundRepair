import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, ReLU, Dense, Add, Flatten
from tensorflow.keras.models import Model

class DL_Model(Model):
    def __init__(self):
        super(DL_Model, self).__init__()
        # First block
        self.conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same')
        self.bn1 = BatchNormalization(axis=-1)
        self.relu1 = ReLU()

        # Second block
        self.conv2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same')
        self.bn2 = BatchNormalization(axis=-1)
        self.relu2 = ReLU()

        # Third block
        self.conv3 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same')
        self.bn3 = BatchNormalization(axis=-1)
        self.relu3 = ReLU()

        # Parallel branch
        self.conv_parallel = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same')

        # Add layer
        self.add = Add()

        # Fully connected layers
        self.flatten = Flatten()
        self.dense1 = Dense(units=128, activation='relu')
        self.dense2 = Dense(units=10, activation='softmax')

    def call(self, inputs):
        # Process through three blocks
        x1 = self.conv1(inputs)
        x1 = self.bn1(x1)
        x1 = self.relu1(x1)

        x2 = self.conv2(x1)
        x2 = self.bn2(x2)
        x2 = self.relu2(x2)

        x3 = self.conv3(x2)
        x3 = self.bn3(x3)
        x3 = self.relu3(x3)

        # Process through parallel branch
        x_parallel = self.conv_parallel(inputs)

        # Add all outputs
        x = self.add([x1, x2, x3, x_parallel])

        # Fully connected layers
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dense2(x)

        return x

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.keras.Input(shape=(32, 32, 3))
    output_tensor = model(input_tensor)
    model = Model(inputs=input_tensor, outputs=output_tensor)
    model.summary()