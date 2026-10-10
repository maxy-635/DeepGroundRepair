import tensorflow as tf
from tensorflow.keras.layers import Layer, Input, AveragePooling2D, Flatten, Dropout, Concatenate, Dense, Reshape, Conv2D, Add, Multiply, Lambda
from tensorflow.keras.models import Model

class DL_Model(Model):
    def __init__(self):
        super(DL_Model, self).__init__()
        # Block 1: Three parallel average pooling paths
        self.pool1x1 = AveragePooling2D(pool_size=(1, 1), strides=(1, 1), padding='valid')
        self.pool2x2 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')
        self.pool4x4 = AveragePooling2D(pool_size=(4, 4), strides=(4, 4), padding='valid')

        # Flatten and dropout layers for each path
        self.flatten1 = Flatten()
        self.flatten2 = Flatten()
        self.flatten3 = Flatten()
        self.dropout1 = Dropout(rate=0.5)
        self.dropout2 = Dropout(rate=0.5)
        self.dropout3 = Dropout(rate=0.5)

        # Concatenate layer for block 1
        self.concat1 = Concatenate(axis=-1)

        # Fully connected layer and reshape for transition
        self.dense_transition = Dense(units=128, activation='relu')
        self.reshape = Reshape(target_shape=(4, 4, 8))  # Assuming input is 28x28 MNIST images

        # Block 2: Four branch connections
        # Branch 1: 1x1 convolution
        self.conv1x1_1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')

        # Branch 2: <1x1 convolution, 3x3 convolution>
        self.conv1x1_2 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')
        self.conv3x3_2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')

        # Branch 3: <1x1 convolution, 3x3 convolution, 3x3 convolution>
        self.conv1x1_3 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')
        self.conv3x3_3a = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')
        self.conv3x3_3b = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')

        # Branch 4: <average pooling, 1x1 convolution>
        self.pool_branch = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')
        self.conv1x1_4 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')

        # Concatenate layer for block 2
        self.concat2 = Concatenate(axis=-1)

        # Final classification layers
        self.dense1 = Dense(units=128, activation='relu')
        self.dense2 = Dense(units=10, activation='softmax')

    def call(self, inputs):
        # Block 1 processing
        # Path 1: 1x1 pooling
        x1 = self.pool1x1(inputs)
        x1 = self.flatten1(x1)
        x1 = self.dropout1(x1)

        # Path 2: 2x2 pooling
        x2 = self.pool2x2(inputs)
        x2 = self.flatten2(x2)
        x2 = self.dropout2(x2)

        # Path 3: 4x4 pooling
        x3 = self.pool4x4(inputs)
        x3 = self.flatten3(x3)
        x3 = self.dropout3(x3)

        # Concatenate all paths
        block1_output = self.concat1([x1, x2, x3])

        # Transition to block 2
        block1_output = self.dense_transition(block1_output)
        block1_output = self.reshape(block1_output)

        # Block 2 processing
        # Branch 1: 1x1 convolution
        branch1 = self.conv1x1_1(block1_output)

        # Branch 2: <1x1 convolution, 3x3 convolution>
        branch2 = self.conv1x1_2(block1_output)
        branch2 = self.conv3x3_2(branch2)

        # Branch 3: <1x1 convolution, 3x3 convolution, 3x3 convolution>
        branch3 = self.conv1x1_3(block1_output)
        branch3 = self.conv3x3_3a(branch3)
        branch3 = self.conv3x3_3b(branch3)

        # Branch 4: <average pooling, 1x1 convolution>
        branch4 = self.pool_branch(block1_output)
        branch4 = self.conv1x1_4(branch4)

        # Concatenate all branches
        block2_output = self.concat2([branch1, branch2, branch3, branch4])

        # Flatten for final classification
        block2_output = tf.keras.layers.Flatten()(block2_output)

        # Final classification layers
        output = self.dense1(block2_output)
        output = self.dense2(output)

        return output

if __name__ == "__main__":
    # Create a dummy input tensor with shape (batch_size, height, width, channels)
    # MNIST images are 28x28 grayscale, so channels=1
    input_tensor = tf.random.normal(shape=(1, 28, 28, 1))

    # Instantiate the model
    model = DL_Model()

    # Get the output tensor
    output_tensor = model(input_tensor)

    print("Input shape:", input_tensor.shape)
    print("Output shape:", output_tensor.shape)