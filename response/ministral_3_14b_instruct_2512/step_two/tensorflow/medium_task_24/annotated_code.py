import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dropout, Concatenate, Dense, Flatten, BatchNormalization, ReLU, Activation
from tensorflow.keras.models import Model

class DL_Model(Model):
    def __init__(self):
        super(DL_Model, self).__init__()
        # Branch 1: 1x1 conv + 3x3 conv
        self.branch1_conv1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')
        self.branch1_conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')
        self.branch1_dropout = Dropout(rate=0.2)

        # Branch 2: 1x1 conv + 1x7 conv + 7x1 conv + 3x3 conv
        self.branch2_conv1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')
        self.branch2_conv2 = Conv2D(filters=64, kernel_size=(1, 7), strides=(1, 1), padding='same', activation='relu')
        self.branch2_conv3 = Conv2D(filters=64, kernel_size=(7, 1), strides=(1, 1), padding='same', activation='relu')
        self.branch2_conv4 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')
        self.branch2_dropout = Dropout(rate=0.2)

        # Branch 3: Max pooling
        self.branch3_pool = MaxPool2D(pool_size='mask', strides='mask', padding='mask')
        self.branch3_conv = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')
        self.branch3_dropout = Dropout(rate=0.2)

        # Concatenate and fully connected layers
        self.concatenate = Concatenate(axis=-1)
        self.flatten = Flatten()
        self.fc1 = Dense(units=512, activation='relu')
        self.fc2 = Dense(units=256, activation='relu')
        self.fc3 = Dense(units=10, activation='softmax')

    def call(self, inputs):
        # the initial shape of inputs is: (1, 32, 32, 3)
        # Branch 1
        x1 = self.branch1_conv1(inputs)  # output tensor shape: (1, 32, 32, 64)
        x1 = self.branch1_conv2(x1)  # output tensor shape: (1, 32, 32, 64)
        x1 = self.branch1_dropout(x1)  # output tensor shape: (1, 32, 32, 64)

        # Branch 2
        x2 = self.branch2_conv1(inputs)  # output tensor shape: (1, 32, 32, 64)
        x2 = self.branch2_conv2(x2)  # output tensor shape: (1, 32, 32, 64)
        x2 = self.branch2_conv3(x2)  # output tensor shape: (1, 32, 32, 64)
        x2 = self.branch2_conv4(x2)  # output tensor shape: (1, 32, 32, 64)
        x2 = self.branch2_dropout(x2)  # output tensor shape: (1, 32, 32, 64)

        # Branch 3
        x3 = self.branch3_pool(inputs)  # output tensor shape of this layer should be (1, 32, 32, 3)
        x3 = self.branch3_conv(x3)
        x3 = self.branch3_dropout(x3)

        # Concatenate all branches
        x = self.concatenate([x1, x2, x3])

        # Fully connected layers
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)

        return x

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal(shape=(1, 32, 32, 3))  # CIFAR-10 input shape
    output_tensor = model(input_tensor)
    print("Model output shape:", output_tensor.shape)