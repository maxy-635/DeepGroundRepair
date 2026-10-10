import tensorflow as tf
from tensorflow.keras.layers import AveragePooling2D, Flatten, Concatenate, Dense, Reshape, Conv2D, Dropout

class DL_Model(tf.keras.Model):
    def __init__(self):
        super(DL_Model, self).__init__()

        # First Block
        self.avg_pool1 = AveragePooling2D(pool_size=(1, 1), strides=(1, 1), padding='valid')
        self.avg_pool2 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')
        self.avg_pool3 = AveragePooling2D(pool_size=(4, 4), strides=(4, 4), padding='valid')
        self.flatten = Flatten()
        self.concat = Concatenate(axis=-1)
        self.dense1 = Dense(units=128, activation='relu')
        self.reshape = Reshape((1, 128))  # Reshape to 4D tensor

        # Second Block
        self.conv2d_path1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='valid', activation='relu')
        self.dropout1 = Dropout(rate=0.2)

        self.conv2d_path2_1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu')
        self.conv2d_path2_2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu')
        self.conv2d_path2_pre = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='valid', activation='relu')
        self.dropout2 = Dropout(rate=0.2)

        self.conv2d_path3_1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu')
        self.conv2d_path3_pre = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='valid', activation='relu')
        self.dropout3 = Dropout(rate=0.2)

        self.avg_pool_path4 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')
        self.conv2d_path4 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='valid', activation='relu')
        self.dropout4 = Dropout(rate=0.2)

        self.concat_paths = Concatenate(axis=-1)

        # Classification Layers
        self.dense2 = Dense(units=64, activation='relu')
        self.dense3 = Dense(units=10, activation='softmax')

    def call(self, inputs):
        # the initial shape of inputs is: (1, 28, 28, 1)
        # First Block
        x = self.avg_pool1(inputs)  # output tensor shape: (1, 28, 28, 1)
        x = self.avg_pool2(x)  # output tensor shape: (1, 14, 14, 1)
        x = self.avg_pool3(x)  # output tensor shape: (1, 3, 3, 1)
        x = self.flatten(x)  # output tensor shape: (1, 9)
        x = self.concat(x)
        x = self.dense1(x)
        x = self.reshape(x)

        # Second Block
        x1 = self.conv2d_path1(x)
        x1 = self.dropout1(x1)

        x2 = self.conv2d_path2_pre(x)
        x2 = self.conv2d_path2_1(x2)
        x2 = self.conv2d_path2_2(x2)
        x2 = self.dropout2(x2)

        x3 = self.conv2d_path3_pre(x)
        x3 = self.conv2d_path3_1(x3)
        x3 = self.dropout3(x3)

        x4 = self.avg_pool_path4(x)
        x4 = self.conv2d_path4(x4)
        x4 = self.dropout4(x4)

        x = self.concat_paths([x1, x2, x3, x4])

        # Classification Layers
        x = self.dense2(x)
        x = self.dense3(x)

        return x


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal((1, 28, 28, 1))  # Dummy input tensor with shape (batch_size, height, width, channels)
    output_tensor = model(input_tensor)
    print("Output shape:", output_tensor.shape)