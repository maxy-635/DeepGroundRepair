import tensorflow as tf
from tensorflow.keras.layers import AveragePooling2D, Flatten, Dropout, Concatenate, Dense, Reshape, Conv2D, GlobalAveragePooling2D

class DL_Model(tf.keras.Model):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Block 1
        self.avg_pool_1x1 = AveragePooling2D(pool_size=(1, 1), strides=(1, 1), padding='valid')
        self.avg_pool_2x2 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')
        self.avg_pool_4x4 = AveragePooling2D(pool_size=(4, 4), strides=(4, 4), padding='valid')
        self.flatten_1 = Flatten()
        self.flatten_2 = Flatten()
        self.flatten_3 = Flatten()
        self.dropout_1 = Dropout(rate=0.25)
        self.dropout_2 = Dropout(rate=0.25)
        self.dropout_3 = Dropout(rate=0.25)
        self.concat_1 = Concatenate(axis=-1)

        # Between Block 1 and Block 2
        self.dense_1 = Dense(units=128, activation='relu')
        self.reshape_1 = Reshape((1, 1, 128))  # Reshape to 4D tensor

        # Block 2
        self.conv2d_1x1_1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='valid', activation='relu')
        self.conv2d_1x1_2 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='valid', activation='relu')
        self.conv2d_3x3_1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu')
        self.conv2d_3x3_2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu')
        self.conv2d_3x3_3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu')
        self.avg_pool_2 = GlobalAveragePooling2D(keepdims=False)
        self.conv2d_1x1_3 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='valid', activation='relu')
        self.concat_2 = Concatenate(axis=-1)

        # Final Classification
        self.dense_2 = Dense(units=128, activation='relu')
        self.dense_3 = Dense(units=10, activation='softmax')  # Assuming 10 classes for MNIST

    def call(self, inputs):
        # Block 1
        x1 = self.avg_pool_1x1(inputs)
        x2 = self.avg_pool_2x2(inputs)
        x3 = self.avg_pool_4x4(inputs)

        x1 = self.flatten_1(x1)
        x2 = self.flatten_2(x2)
        x3 = self.flatten_3(x3)

        x1 = self.dropout_1(x1)
        x2 = self.dropout_2(x2)
        x3 = self.dropout_3(x3)

        x = self.concat_1([x1, x2, x3])

        # Between Block 1 and Block 2
        x = self.dense_1(x)
        x = self.reshape_1(x)

        # Block 2
        x1 = self.conv2d_1x1_1(x)
        x2 = self.conv2d_1x1_2(x)
        x3 = self.conv2d_3x3_1(x)
        x4 = self.conv2d_3x3_2(x)
        x5 = self.conv2d_3x3_3(x)
        x6 = self.avg_pool_2(x)
        x7 = self.conv2d_1x1_3(x6)

        x = self.concat_2([x1, x2, x3, x4, x5, x7])

        # Final Classification
        x = self.dense_2(x)
        x = self.dense_3(x)

        return x


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal((1, 28, 28, 1))  # Dummy input for MNIST (batch_size, height, width, channels)
    output_tensor = model(input_tensor)
    print("Output shape:", output_tensor.shape)