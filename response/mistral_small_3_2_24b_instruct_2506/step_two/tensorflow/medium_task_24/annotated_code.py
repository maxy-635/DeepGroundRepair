import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dropout, Concatenate, Dense
from tensorflow.keras.models import Model

class DL_Model(Model):
    def __init__(self):
        super(DL_Model, self).__init__()
        # Branch 1: 1x1 convolution followed by 3x3 convolution
        self.branch1_conv1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')
        self.branch1_conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')
        self.branch1_dropout = Dropout(rate=0.5)

        # Branch 2: 1x1 convolution followed by 1x7, 7x1, and 3x3 convolutions
        self.branch2_conv1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')
        self.branch2_conv2 = Conv2D(filters=64, kernel_size=(1, 7), strides=(1, 1), padding='same', activation='relu')
        self.branch2_conv3 = Conv2D(filters=64, kernel_size=(7, 1), strides=(1, 1), padding='same', activation='relu')
        self.branch2_conv4 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')
        self.branch2_dropout = Dropout(rate=0.5)

        # Branch 3: Max pooling
        self.branch3_pool = MaxPool2D(pool_size=(3, 3), strides=(1, 1), padding='same')
        self.branch3_dropout = Dropout(rate=0.5)

        # Concatenation and fully connected layers
        self.concatenate = Concatenate(axis=-1)
        self.flatten = tf.keras.layers.Flatten()
        self.fc1 = Dense(units=512, activation='relu')
        self.fc2 = Dense(units=256, activation='relu')
        self.fc3 = Dense(units=10, activation='softmax')

    def call(self, inputs):
        # Process through branch 1
        branch1 = self.branch1_conv1(inputs)
        branch1 = self.branch1_conv2(branch1)
        branch1 = self.branch1_dropout(branch1)

        # Process through branch 2
        branch2 = self.branch2_conv1(inputs)
        branch2 = self.branch2_conv2(branch2)
        branch2 = self.branch2_conv3(branch2)
        branch2 = self.branch2_conv4(branch2)
        branch2 = self.branch2_dropout(branch2)

        # Process through branch 3
        branch3 = self.branch3_pool(inputs)
        branch3 = self.branch3_dropout(branch3)

        # Concatenate all branches
        concatenated = self.concatenate([branch1, branch2, branch3])

        # Flatten and pass through fully connected layers
        x = self.flatten(concatenated)
        x = self.fc1(x)
        x = self.fc2(x)
        output = self.fc3(x)

        return output

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.keras.Input(shape=(32, 32, 3))  # CIFAR-10 input shape
    output_tensor = model(input_tensor)
    print("Model created successfully with output shape:", output_tensor.shape)