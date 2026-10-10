import tensorflow as tf
from tensorflow.keras.layers import AveragePooling2D, Flatten, Dropout, Concatenate, Dense, Reshape, Conv2D
from tensorflow.keras.models import Model

class DL_Model(Model):
    def __init__(self):
        super(DL_Model, self).__init__()
        # Block 1: Three parallel paths with different pooling scales
        self.avg_pool_1x1 = AveragePooling2D(pool_size=(1, 1), strides=(1, 1), padding='valid')
        self.avg_pool_2x2 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')
        self.avg_pool_4x4 = AveragePooling2D(pool_size=(4, 4), strides=(4, 4), padding='valid')

        self.flatten_1 = Flatten()
        self.flatten_2 = Flatten()
        self.flatten_3 = Flatten()

        self.dropout_1 = Dropout(rate=0.5)
        self.dropout_2 = Dropout(rate=0.5)
        self.dropout_3 = Dropout(rate=0.5)

        self.concatenate_1 = Concatenate(axis=-1)

        # Fully connected layer and reshaping
        self.dense_1 = Dense(units=128, activation='relu')
        self.reshape_1 = Reshape(target_shape=(4, 4, 8))  # Example shape, adjust as needed

        # Block 2: Multiple branch connections
        self.conv_1x1_1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')
        self.conv_1x1_2 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')
        self.conv_3x3_1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')
        self.conv_3x3_2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')
        self.avg_pool_2 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')
        self.conv_1x1_3 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')

        self.concatenate_2 = Concatenate(axis=-1)

        # Final classification layers
        self.dense_2 = Dense(units=64, activation='relu')
        self.dense_3 = Dense(units=10, activation='softmax')  # Assuming 10 classes for MNIST

    def call(self, inputs):
        # Block 1
        path1 = self.avg_pool_1x1(inputs)
        path1 = self.flatten_1(path1)
        path1 = self.dropout_1(path1)

        path2 = self.avg_pool_2x2(inputs)
        path2 = self.flatten_2(path2)
        path2 = self.dropout_2(path2)

        path3 = self.avg_pool_4x4(inputs)
        path3 = self.flatten_3(path3)
        path3 = self.dropout_3(path3)

        block1_output = self.concatenate_1([path1, path2, path3])

        # Fully connected layer and reshaping
        block1_output = self.dense_1(block1_output)
        block1_output = self.reshape_1(block1_output)

        # Block 2
        branch1 = self.conv_1x1_1(block1_output)
        branch2 = self.conv_1x1_2(block1_output)
        branch2 = self.conv_3x3_1(branch2)
        branch3 = self.conv_1x1_2(block1_output)
        branch3 = self.conv_3x3_1(branch3)
        branch3 = self.conv_3x3_2(branch3)
        branch4 = self.avg_pool_2(block1_output)
        branch4 = self.conv_1x1_3(branch4)

        block2_output = self.concatenate_2([branch1, branch2, branch3, branch4])

        # Final classification
        output = self.dense_2(block2_output)
        output = self.dense_3(output)

        return output

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal(shape=(1, 28, 28, 1))  # MNIST image shape: (28, 28, 1)
    output_tensor = model(input_tensor)
    print("Input shape:", input_tensor.shape)
    print("Output shape:", output_tensor.shape)