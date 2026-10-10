import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, ReLU, Concatenate, Flatten, Dense
from tensorflow.keras.models import Model

class DL_Model(Model):
    def __init__(self):
        super(DL_Model, self).__init__()
        # Block 1
        self.conv1_1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same')
        self.bn1_1 = BatchNormalization(axis=-1)
        self.relu1_1 = ReLU()

        # Block 2
        self.conv2_1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same')
        self.bn2_1 = BatchNormalization(axis=-1)
        self.relu2_1 = ReLU()

        # Block 3
        self.conv3_1 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same')
        self.bn3_1 = BatchNormalization(axis=-1)
        self.relu3_1 = ReLU()

        # Fully connected layers
        self.flatten = Flatten()
        self.dense1 = Dense(units=256, activation='relu')
        self.dense2 = Dense(units=10, activation='softmax')

    def call(self, inputs):
        # Block 1
        x = self.conv1_1(inputs)
        x = self.bn1_1(x)
        x = self.relu1_1(x)
        block1_output = x

        # Block 2
        x = self.conv2_1(block1_output)
        x = self.bn2_1(x)
        x = self.relu2_1(x)
        block2_output = Concatenate(axis=-1)([block1_output, x])

        # Block 3
        x = self.conv3_1(block2_output)
        x = self.bn3_1(x)
        x = self.relu3_1(x)
        block3_output = Concatenate(axis=-1)([block2_output, x])

        # Fully connected layers
        x = self.flatten(block3_output)
        x = self.dense1(x)
        outputs = self.dense2(x)

        return outputs

if __name__ == "__main__":
    model = DL_Model()
    # CIFAR-10 images are 32x32 with 3 channels
    input_tensor = tf.random.normal(shape=(1, 32, 32, 3))
    output_tensor = model(input_tensor)
    print("Input shape:", input_tensor.shape)
    print("Output shape:", output_tensor.shape)