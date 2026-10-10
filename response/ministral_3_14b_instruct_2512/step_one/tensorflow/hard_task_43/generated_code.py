import tensorflow as tf
from tensorflow.keras.layers import Conv2D, AveragePooling2D, Flatten, Concatenate, Dense, Reshape, Lambda, Add, Activation, BatchNormalization
from tensorflow.keras.models import Model

class DL_Model(Model):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Block 1: Three parallel average pooling paths
        self.avg_pool_1x1 = AveragePooling2D(pool_size=(1, 1), strides=(1, 1), padding='valid')
        self.avg_pool_2x2 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')
        self.avg_pool_4x4 = AveragePooling2D(pool_size=(4, 4), strides=(4, 4), padding='valid')

        # Flatten and concatenate paths
        self.flatten_1 = Flatten()
        self.flatten_2 = Flatten()
        self.flatten_3 = Flatten()
        self.concatenate = Concatenate(axis=-1)

        # Fully connected layer between blocks
        self.dense_between = Dense(128, activation='relu')

        # Reshape for Block 2
        self.reshape = Reshape((4, 4, 128))

        # Block 2: Three branches for feature extraction
        # Branch 1: 1x1 conv + 3x3 conv
        self.conv1_1x1 = Conv2D(64, (1, 1), strides=(1, 1), padding='same', activation='relu')
        self.conv1_3x3 = Conv2D(64, (3, 3), strides=(1, 1), padding='same', activation='relu')

        # Branch 2: 1x1 conv + 1x7 conv + 7x1 conv + 3x3 conv
        self.conv2_1x1 = Conv2D(64, (1, 1), strides=(1, 1), padding='same', activation='relu')
        self.conv2_1x7 = Conv2D(64, (1, 7), strides=(1, 1), padding='same', activation='relu')
        self.conv2_7x1 = Conv2D(64, (7, 1), strides=(1, 1), padding='same', activation='relu')
        self.conv2_3x3 = Conv2D(64, (3, 3), strides=(1, 1), padding='same', activation='relu')

        # Branch 3: Average pooling
        self.avg_pool_branch = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')

        # Final classification layers
        self.dense_final1 = Dense(256, activation='relu')
        self.dense_final2 = Dense(10, activation='softmax')

    def call(self, inputs):
        # Block 1 processing
        path1 = self.avg_pool_1x1(inputs)
        path2 = self.avg_pool_2x2(inputs)
        path3 = self.avg_pool_4x4(inputs)

        # Flatten and concatenate
        path1 = self.flatten_1(path1)
        path2 = self.flatten_2(path2)
        path3 = self.flatten_3(path3)

        combined = self.concatenate([path1, path2, path3])

        # Fully connected layer and reshape
        fc_output = self.dense_between(combined)
        reshaped = self.reshape(fc_output)

        # Block 2 processing
        # Branch 1
        branch1 = self.conv1_1x1(reshaped)
        branch1 = self.conv1_3x3(branch1)

        # Branch 2
        branch2 = self.conv2_1x1(reshaped)
        branch2 = self.conv2_1x7(branch2)
        branch2 = self.conv2_7x1(branch2)
        branch2 = self.conv2_3x3(branch2)

        # Branch 3
        branch3 = self.avg_pool_branch(reshaped)

        # Concatenate all branches
        block2_output = self.concatenate([branch1, branch2, branch3])

        # Flatten and final classification
        flattened = Flatten()(block2_output)
        final_output = self.dense_final1(flattened)
        final_output = self.dense_final2(final_output)

        return final_output

if __name__ == "__main__":
    model = DL_Model()
    # MNIST images are 28x28 with 1 channel
    input_tensor = tf.keras.Input(shape=(28, 28, 1))
    output_tensor = model(input_tensor)
    print("Model created successfully!")
    print("Input shape:", input_tensor.shape)
    print("Output shape:", output_tensor.shape)