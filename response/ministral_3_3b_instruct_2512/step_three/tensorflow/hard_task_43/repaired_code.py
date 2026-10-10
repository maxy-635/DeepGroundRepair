import tensorflow as tf
from tensorflow.keras.layers import AveragePooling2D, Flatten, Concatenate, Dense, Reshape, Conv2D, BatchNormalization, Activation, Dropout

class DL_Model(tf.keras.Model):
    def __init__(self):
        super(DL_Model, self).__init__()
        # Block 1: Three parallel paths with different average pooling layers
        self.pool1x1 = AveragePooling2D(pool_size=(1, 1), strides=(1, 1), padding='valid')
        self.pool2x2 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')
        self.pool4x4 = AveragePooling2D(pool_size=(4, 4), strides=(4, 4), padding='valid')

        # Flatten and concatenate the outputs of the three pooling layers
        self.flatten = [Flatten() for _ in range(3)]
        self.concatenate_block1 = Concatenate(axis=-1)

        # Fully connected layer and reshape for Block 2
        self.fc = Dense(units=128, activation='relu')
        self.reshape = Reshape(target_shape=(4, 4, 1))

        # Block 2: Three branches for feature extraction
        # Branch 1: <1x1 convolution, 3x3 convolution>
        self.conv1x1_3x3_branch1 = tf.keras.Sequential([
            Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu'),
            Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')
        ])

        # Branch 2: <1x1 convolution, 1x7 convolution, 7x1 convolution, 3x3 convolution>
        self.conv1x1_1x7_7x1_3x3_branch2 = tf.keras.Sequential([
            Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu'),
            Conv2D(filters=32, kernel_size=(1, 7), strides=(1, 1), padding='same', activation='relu'),
            Conv2D(filters=32, kernel_size=(7, 1), strides=(1, 1), padding='same', activation='relu'),
            Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')
        ])

        # Branch 3: Average pooling
        self.pool_avg_branch3 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')

        # Concatenate the outputs of the three branches
        self.concatenate_block2 = Concatenate(axis=-1)

        # Fully connected layers for classification
        self.fc_classification1 = Dense(units=128, activation='relu')
        self.fc_classification2 = Dense(units=10, activation='softmax')

    def call(self, inputs):
        # the initial shape of inputs is: (1, 28, 28, 1)
        # Block 1 processing
        pool1x1_output = self.pool1x1(inputs)  # output tensor shape: (1, 28, 28, 1)
        pool2x2_output = self.pool2x2(inputs)  # output tensor shape: (1, 14, 14, 1)
        pool4x4_output = self.pool4x4(inputs)  # output tensor shape: (1, 7, 7, 1)

        flattened_outputs = self.flatten([pool1x1_output, pool2x2_output, pool4x4_output])
        block1_output = self.concatenate_block1(flattened_outputs)

        # Apply fully connected layer and reshape
        block1_output = self.fc(block1_output)
        block1_output = self.reshape(block1_output)

        # Block 2 processing
        branch1_output = self.conv1x1_3x3_branch1(block1_output)
        branch2_output = self.conv1x1_1x7_7x1_3x3_branch2(block1_output)
        branch3_output = self.pool_avg_branch3(block1_output)

        block2_output = self.concatenate_block2([branch1_output, branch2_output, branch3_output])

        # Classification layers
        classification_output = self.fc_classification1(block2_output)
        output = self.fc_classification2(classification_output)

        return output

if __name__ == "__main__":
    # Dummy input tensor with shape (batch_size, height, width, channels)
    input_tensor = tf.random.normal(shape=(1, 28, 28, 1))
    model = DL_Model()
    output_tensor = model(input_tensor)
    print("Input shape:", input_tensor.shape)
    print("Output shape:", output_tensor.shape)