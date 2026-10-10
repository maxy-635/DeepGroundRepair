import tensorflow as tf
from tensorflow.keras import layers

class DL_Model(tf.keras.Model):
    def __init__(self):
        super(DL_Model, self).__init__()
        # Block 1: Three parallel paths with different max pooling scales
        self.pool1x1 = layers.MaxPool2D(pool_size=(1, 1), strides=(1, 1), padding='valid')
        self.pool2x2 = layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid')
        self.pool4x4 = layers.MaxPool2D(pool_size=(4, 4), strides=(4, 4), padding='valid')

        # Dropout layers for regularization
        self.dropout1 = layers.Dropout(rate=0.2)
        self.dropout2 = layers.Dropout(rate=0.2)
        self.dropout3 = layers.Dropout(rate=0.2)

        # Concatenate the outputs of the three paths
        self.concatenate1 = layers.Concatenate(axis=-1)

        # Fully connected layer and reshaping for block 2 input
        self.flatten = layers.Flatten()
        self.dense = layers.Dense(units=1024, activation=None)  # No activation for reshaping
        self.reshape = layers.Reshape(target_shape=(4, 128))  # Adjust units to match expected shape

        # Block 2: Four parallel paths with different convolution and pooling strategies
        # Path 1: Single 1x1 convolution
        self.conv_path1 = layers.Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='valid')

        # Path 2: 1x1 convolution followed by 1x7 and 7x1 convolutions
        self.conv_path2_1x1 = layers.Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='valid')
        self.conv_path2_1x7 = layers.Conv2D(filters=64, kernel_size=(1, 7), strides=(1, 1), padding='valid')
        self.conv_path2_7x1 = layers.Conv2D(filters=64, kernel_size=(7, 1), strides=(1, 1), padding='valid')

        # Path 3: Alternating 7x1 and 1x7 convolutions
        self.conv_path3_1x1 = layers.Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='valid')
        self.conv_path3_7x1 = layers.Conv2D(filters=64, kernel_size=(7, 1), strides=(1, 1), padding='valid')
        self.conv_path3_1x7 = layers.Conv2D(filters=64, kernel_size=(1, 7), strides=(1, 1), padding='valid')

        # Path 4: Average pooling followed by 1x1 convolution
        self.avg_pool_path4 = layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')
        self.conv_path4 = layers.Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='valid')

        # Concatenate the outputs of the four paths
        self.concatenate2 = layers.Concatenate(axis=-1)

        # Final classification layers
        self.fc1 = layers.Dense(units=512, activation='relu')
        self.fc2 = layers.Dense(units=10, activation='softmax')

    def call(self, inputs):
        # Block 1: Three parallel paths
        path1 = self.pool1x1(inputs)
        path2 = self.pool2x2(inputs)
        path3 = self.pool4x4(inputs)

        # Apply dropout and flatten
        path1 = self.dropout1(path1)
        path2 = self.dropout2(path2)
        path3 = self.dropout3(path3)

        # Flatten each path and concatenate
        flattened_path1 = self.flatten(path1)
        flattened_path2 = self.flatten(path2)
        flattened_path3 = self.flatten(path3)

        concatenated_block1 = self.concatenate1([flattened_path1, flattened_path2, flattened_path3])

        # Reshape for block 2
        reshaped_block1 = self.reshape(self.dense(concatenated_block1))

        # Block 2: Four parallel paths
        path1_block2 = self.conv_path1(reshaped_block1)
        path2_block2 = self.conv_path2_1x1(self.conv_path2_1x7(self.conv_path2_7x1(reshaped_block1)))
        path3_block2 = self.conv_path3_1x1(self.conv_path3_7x1(self.conv_path3_1x7(reshaped_block1)))
        path4_block2 = self.conv_path4(self.avg_pool_path4(reshaped_block1))

        # Concatenate the outputs of the four paths
        concatenated_block2 = self.concatenate2([path1_block2, path2_block2, path3_block2, path4_block2])

        # Final classification
        output = self.fc1(concatenated_block2)
        output = self.fc2(output)

        return output

if __name__ == "__main__":
    model = DL_Model()
    # Dummy input tensor with shape (batch_size, height, width, channels)
    dummy_input = tf.random.normal((1, 28, 28, 1))
    output_tensor = model(dummy_input)
    print("Model output shape:", output_tensor.shape)