import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dropout, Flatten, Dense, concatenate, Reshape

class DL_Model(tf.keras.Model):
    def __init__(self):
        super(DL_Model, self).__init__()
        # Branch 1: <1x1 convolution, 3x3 convolution>
        self.branch1_conv1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='valid', activation='relu')
        self.branch1_conv3 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')
        self.branch1_dropout = Dropout(rate=0.2)

        # Branch 2: <1x1 convolution, 1x7 convolution, 7x1 convolution, 3x3 convolution>
        self.branch2_conv1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='valid', activation='relu')
        self.branch2_conv7x1 = Conv2D(filters=32, kernel_size=(1, 7), strides=(1, 1), padding='valid', activation='relu')
        self.branch2_conv1x7 = Conv2D(filters=32, kernel_size=(7, 1), strides=(1, 1), padding='valid', activation='relu')
        self.branch2_conv3 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')
        self.branch2_dropout = Dropout(rate=0.2)

        # Branch 3: <max pooling>
        self.branch3_pool = MaxPool2D(pool_size='mask', strides='mask', padding='mask')

        # Fully connected layers
        self.flatten = Flatten()
        self.fc1 = Dense(units=128, activation='relu')
        self.fc2 = Dense(units=64, activation='relu')
        self.fc3 = Dense(units=10, activation='softmax')

    def call(self, inputs):
        # the initial shape of inputs is: (1, 32, 32, 3)
        # Branch 1 processing
        branch1_conv1 = self.branch1_conv1(inputs)  # output tensor shape: (1, 32, 32, 32)
        branch1_conv3 = self.branch1_conv3(branch1_conv1)  # output tensor shape: (1, 32, 32, 32)
        branch1_dropout = self.branch1_dropout(branch1_conv3)  # output tensor shape: (1, 32, 32, 32)

        # Branch 2 processing
        branch2_conv1 = self.branch2_conv1(inputs)  # output tensor shape: (1, 32, 32, 32)
        branch2_conv7x1 = self.branch2_conv7x1(branch2_conv1)  # output tensor shape: (1, 32, 26, 32)
        branch2_conv1x7 = self.branch2_conv1x7(branch2_conv7x1)  # output tensor shape: (1, 26, 26, 32)
        branch2_conv3 = self.branch2_conv3(branch2_conv1x7)  # output tensor shape: (1, 26, 26, 32)
        branch2_dropout = self.branch2_dropout(branch2_conv3)  # output tensor shape: (1, 26, 26, 32)

        # Branch 3 processing
        branch3_pool = self.branch3_pool(inputs)  # output tensor shape of this layer should be (1, 32, 32, 3)

        # Concatenate all branches
        concatenated = concatenate([branch1_dropout, branch2_dropout, branch3_pool], axis=-1)

        # Fully connected layers
        flattened = self.flatten(concatenated)
        fc1 = self.fc1(flattened)
        fc2 = self.fc2(fc1)
        output = self.fc3(fc2)

        return output

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal(shape=(1, 32, 32, 3))
    output_tensor = model(input_tensor)
    print("Input shape:", input_tensor.shape)
    print("Output shape:", output_tensor.shape)