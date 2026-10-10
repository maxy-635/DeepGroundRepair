import tensorflow as tf
from tensorflow.keras.layers import MaxPool2D, Dropout, Concatenate, Dense, Reshape, Conv2D, AveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input

class DL_Model(Model):
    def __init__(self):
        super(DL_Model, self).__init__()
        # Block 1: Three parallel paths with different max pooling scales
        self.max_pool_1x1 = MaxPool2D(pool_size=(1, 1), strides=(1, 1), padding='same')
        self.max_pool_2x2 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')
        self.max_pool_4x4 = MaxPool2D(pool_size=(4, 4), strides=(4, 4), padding='same')

        self.dropout_1 = Dropout(rate=0.5)
        self.dropout_2 = Dropout(rate=0.5)
        self.dropout_3 = Dropout(rate=0.5)

        self.concatenate_1 = Concatenate(axis=-1)

        # Fully connected layer and reshaping
        self.dense_1 = Dense(units=128, activation='relu')
        self.reshape_1 = Reshape(target_shape=(4, 4, 8))

        # Block 2: Four parallel paths with different convolution and pooling strategies
        # Path 1: Single 1x1 convolution
        self.conv1_1x1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')

        # Path 2: 1x1 convolution followed by 1x7 and 7x1 convolutions
        self.conv2_1x1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')
        self.conv2_1x7 = Conv2D(filters=32, kernel_size=(1, 7), strides=(1, 1), padding='same', activation='relu')
        self.conv2_7x1 = Conv2D(filters=32, kernel_size=(7, 1), strides=(1, 1), padding='same', activation='relu')

        # Path 3: 1x1 convolution followed by alternating 7x1 and 1x7 convolutions
        self.conv3_1x1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')
        self.conv3_7x1 = Conv2D(filters=32, kernel_size=(7, 1), strides=(1, 1), padding='same', activation='relu')
        self.conv3_1x7 = Conv2D(filters=32, kernel_size=(1, 7), strides=(1, 1), padding='same', activation='relu')

        # Path 4: Average pooling followed by 1x1 convolution
        self.avg_pool = AveragePooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')
        self.conv4_1x1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')

        self.concatenate_2 = Concatenate(axis=-1)

        # Final fully connected layers
        self.dense_2 = Dense(units=64, activation='relu')
        self.dense_3 = Dense(units=10, activation='softmax')

    def call(self, inputs):
        # Block 1: Three parallel paths
        path1 = self.max_pool_1x1(inputs)
        path1 = tf.keras.layers.Flatten()(path1)
        path1 = self.dropout_1(path1)

        path2 = self.max_pool_2x2(inputs)
        path2 = tf.keras.layers.Flatten()(path2)
        path2 = self.dropout_2(path2)

        path3 = self.max_pool_4x4(inputs)
        path3 = tf.keras.layers.Flatten()(path3)
        path3 = self.dropout_3(path3)

        # Concatenate the paths
        concatenated = self.concatenate_1([path1, path2, path3])

        # Fully connected layer and reshaping
        fc_output = self.dense_1(concatenated)
        reshaped = self.reshape_1(fc_output)

        # Block 2: Four parallel paths
        # Path 1
        path1 = self.conv1_1x1(reshaped)

        # Path 2
        path2 = self.conv2_1x1(reshaped)
        path2 = self.conv2_1x7(path2)
        path2 = self.conv2_7x1(path2)

        # Path 3
        path3 = self.conv3_1x1(reshaped)
        path3 = self.conv3_7x1(path3)
        path3 = self.conv3_1x7(path3)

        # Path 4
        path4 = self.avg_pool(reshaped)
        path4 = self.conv4_1x1(path4)

        # Concatenate the paths
        concatenated = self.concatenate_2([path1, path2, path3, path4])

        # Final fully connected layers
        output = self.dense_2(concatenated)
        output = tf.keras.layers.Flatten()(output)
        output = self.dense_3(output)

        return output

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = Input(shape=(28, 28, 1))
    output_tensor = model(input_tensor)
    print(output_tensor.shape)