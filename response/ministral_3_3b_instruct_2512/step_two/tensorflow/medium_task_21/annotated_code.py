import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dropout, concatenate, Dense, Flatten, GlobalAveragePooling2D

class DL_Model(tf.keras.Model):
    def __init__(self):
        super(DL_Model, self).__init__()
        # Branch 1: 1x1 convolution
        self.branch1 = Conv2D(filters=64, kernel_size='mask', activation='relu', strides='mask', padding='mask')

        # Branch 2: 1x1 convolution followed by 3x3 convolution
        self.branch2 = tf.keras.Sequential([
            Conv2D(64, kernel_size=(1, 1), activation='relu'),
            Conv2D(64, kernel_size=(3, 3), activation='relu')
        ])

        # Branch 3: 1x1 convolution followed by two consecutive 3x3 convolutions
        self.branch3 = tf.keras.Sequential([
            Conv2D(64, kernel_size=(1, 1), activation='relu'),
            Conv2D(64, kernel_size=(3, 3), activation='relu'),
            Conv2D(64, kernel_size=(3, 3), activation='relu')
        ])

        # Branch 4: Average pooling followed by 1x1 convolution
        self.branch4 = tf.keras.Sequential([
            MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid'),
            Conv2D(64, kernel_size=(1, 1), activation='relu')
        ])

        # Dropout layers for each branch
        self.dropout1 = Dropout(0.5)
        self.dropout2 = Dropout(0.5)
        self.dropout3 = Dropout(0.5)
        self.dropout4 = Dropout(0.5)

        # Fully connected layers
        self.dense1 = Dense(128, activation='relu')
        self.dense2 = Dense(64, activation='relu')
        self.dense3 = Dense(10, activation='relu')  # CIFAR-10 has 10 classes

    def call(self, inputs):
        # the initial shape of inputs is: (1, 32, 32, 3)
        # Process each branch
        branch1_output = self.branch1(inputs)  # output tensor shape of this layer should be (1, 30, 30, 64)
        branch2_output = self.branch2(inputs)  # output tensor shape: (1, 30, 30, 64)
        branch3_output = self.branch3(inputs)  # output tensor shape: (1, 28, 28, 64)
        branch4_output = self.branch4(inputs)  # output tensor shape: (1, 16, 16, 64)

        # Apply dropout to each branch
        branch1_output = self.dropout1(branch1_output)
        branch2_output = self.dropout2(branch2_output)  # output tensor shape: (1, 30, 30, 64)
        branch2_output = self.dropout2(branch2_output)  # output tensor shape: (1, 30, 30, 64)
        branch3_output = self.dropout3(branch3_output)  # output tensor shape: (1, 28, 28, 64)
        branch4_output = self.dropout4(branch4_output)  # output tensor shape: (1, 16, 16, 64)

        # Concatenate all branches
        concatenated_output = concatenate([branch1_output, branch2_output, branch3_output, branch4_output], axis=-1)

        # Flatten and pass through dense layers
        flattened_output = Flatten()(concatenated_output)
        output = self.dense1(flattened_output)
        output = self.dense2(output)
        output = self.dense3(output)

        return output

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal(shape=(1, 32, 32, 3))  # Example input shape for CIFAR-10
    output_tensor = model(input_tensor)
    print(output_tensor.shape)