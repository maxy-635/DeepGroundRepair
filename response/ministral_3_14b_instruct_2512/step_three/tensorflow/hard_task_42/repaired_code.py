import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dropout, Concatenate, Dense, Reshape, AveragePooling2D, Lambda
from tensorflow.keras.models import Model

class DL_Model(Model):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Block 1: Three parallel max pooling paths
        self.max_pool_1x1 = MaxPool2D(pool_size=(1, 1), strides=(1, 1), padding='same')
        self.max_pool_2x2 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')
        self.max_pool_4x4 = MaxPool2D(pool_size=(4, 4), strides=(4, 4), padding='same')

        self.flatten = Flatten()
        self.dropout = Dropout(0.5)

        # Connection between blocks
        self.dense_transform = Dense(1024, activation='relu')  # Increased output size to match reshaping
        self.reshape = Reshape((4, 4, 64))  # Assuming output channels after block 1

        # Block 2: Four parallel paths with different convolution strategies
        # Path 1: Single 1x1 convolution
        self.conv1_path1 = Conv2D(16, (1, 1), strides=(1, 1), padding='same', activation='relu')

        # Path 2: 1x1 + 1x7 + 7x1 convolutions
        self.conv1_path2 = Conv2D(16, (1, 1), strides=(1, 1), padding='same', activation='relu')
        self.conv2_path2 = Conv2D(16, (1, 7), strides=(1, 1), padding='same', activation='relu')
        self.conv3_path2 = Conv2D(16, (7, 1), strides=(1, 1), padding='same', activation='relu')

        # Path 3: 1x1 + 7x1 + 1x7 convolutions
        self.conv1_path3 = Conv2D(16, (1, 1), strides=(1, 1), padding='same', activation='relu')
        self.conv2_path3 = Conv2D(16, (7, 1), strides=(1, 1), padding='same', activation='relu')
        self.conv3_path3 = Conv2D(16, (1, 7), strides=(1, 1), padding='same', activation='relu')

        # Path 4: Average pooling + 1x1 convolution
        self.avg_pool_path4 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')
        self.conv_path4 = Conv2D(16, (1, 1), strides=(1, 1), padding='same', activation='relu')

        # Final classification layers
        self.fc1 = Dense(128, activation='relu')
        self.fc2 = Dense(10, activation='softmax')  # 10 classes for MNIST

    def call(self, inputs):
        # Block 1 processing
        # Path 1: 1x1 max pooling
        x1 = self.max_pool_1x1(inputs)
        x1 = self.flatten(x1)
        x1 = self.dropout(x1)

        # Path 2: 2x2 max pooling
        x2 = self.max_pool_2x2(inputs)
        x2 = self.flatten(x2)
        x2 = self.dropout(x2)

        # Path 3: 4x4 max pooling
        x3 = self.max_pool_4x4(inputs)
        x3 = self.flatten(x3)
        x3 = self.dropout(x3)

        # Concatenate all paths from Block 1
        block1_output = Concatenate(axis=1)([x1, x2, x3])

        # Transform for Block 2
        block1_output = self.dense_transform(block1_output)
        block1_output = self.reshape(block1_output)

        # Block 2 processing
        # Path 1: Single 1x1 convolution
        path1 = self.conv1_path1(block1_output)

        # Path 2: 1x1 + 1x7 + 7x1 convolutions
        path2 = self.conv1_path2(block1_output)
        path2 = self.conv2_path2(path2)
        path2 = self.conv3_path2(path2)

        # Path 3: 1x1 + 7x1 + 1x7 convolutions
        path3 = self.conv1_path3(block1_output)
        path3 = self.conv2_path3(path3)
        path3 = self.conv3_path3(path3)

        # Path 4: Average pooling + 1x1 convolution
        path4 = self.avg_pool_path4(block1_output)
        path4 = self.conv_path4(path4)

        # Concatenate all paths from Block 2
        block2_output = Concatenate(axis=-1)([path1, path2, path3, path4])

        # Final classification
        x = self.fc1(block2_output)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = self.fc2(x)

        return x

if __name__ == "__main__":
    model = DL_Model()

    # MNIST input shape: (28, 28, 1)
    input_tensor = tf.keras.Input(shape=(28, 28, 1))
    output_tensor = model(input_tensor)

    # Print model summary
    model.summary()