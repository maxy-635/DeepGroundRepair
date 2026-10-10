import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Dense, Dropout, AveragePooling2D, Concatenate, Flatten, BatchNormalization, Activation
from tensorflow.keras.models import Model

class DL_Model(Model):
    def __init__(self):
        super(DL_Model, self).__init__()
        # Branch 1: 1x1 convolution
        self.branch1_conv = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')
        self.branch1_dropout = Dropout(rate=0.2)

        # Branch 2: 1x1 convolution followed by 3x3 convolution
        self.branch2_conv1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')
        self.branch2_conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')
        self.branch2_dropout = Dropout(rate=0.2)

        # Branch 3: 1x1 convolution followed by two consecutive 3x3 convolutions
        self.branch3_conv1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')
        self.branch3_conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')
        self.branch3_conv3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')
        self.branch3_dropout = Dropout(rate=0.2)

        # Branch 4: Average pooling followed by 1x1 convolution
        self.branch4_pool = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')
        self.branch4_conv = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')
        self.branch4_dropout = Dropout(rate=0.2)

        # Fully connected layers
        self.fc1 = Dense(units=512, activation='relu')
        self.fc2 = Dense(units=256, activation='relu')
        self.fc3 = Dense(units=10, activation='softmax')  # CIFAR-10 has 10 classes

    def call(self, inputs):
        # Branch 1
        x1 = self.branch1_conv(inputs)
        x1 = self.branch1_dropout(x1)

        # Branch 2
        x2 = self.branch2_conv1(inputs)
        x2 = self.branch2_conv2(x2)
        x2 = self.branch2_dropout(x2)

        # Branch 3
        x3 = self.branch3_conv1(inputs)
        x3 = self.branch3_conv2(x3)
        x3 = self.branch3_conv3(x3)
        x3 = self.branch3_dropout(x3)

        # Branch 4
        x4 = self.branch4_pool(inputs)
        x4 = self.branch4_conv(x4)
        x4 = self.branch4_dropout(x4)

        # Concatenate all branches
        concatenated = Concatenate(axis=-1)([x1, x2, x3, x4])

        # Flatten and pass through fully connected layers
        flattened = Flatten()(concatenated)
        x = self.fc1(flattened)
        x = self.fc2(x)
        outputs = self.fc3(x)

        return outputs

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal(shape=(1, 32, 32, 3))  # CIFAR-10 input shape
    output_tensor = model(input_tensor)
    print("Model output shape:", output_tensor.shape)