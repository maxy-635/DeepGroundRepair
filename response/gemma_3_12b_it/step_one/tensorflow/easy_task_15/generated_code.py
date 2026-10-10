import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Conv1D, AveragePooling2D, Dropout, GlobalAveragePooling2D, Flatten, Dense

class DL_Model(tf.keras.Model):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Specialized block
        self.conv2d_1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')
        self.conv1d_1 = Conv1D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')
        self.conv1d_2 = Conv1D(filters=8, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')
        self.avg_pool = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')
        self.dropout = Dropout(rate=0.25)

        # Two consecutive blocks
        self.block_1 = tf.keras.Sequential([
            self.conv2d_1,
            self.conv1d_1,
            self.conv1d_2,
            self.avg_pool,
            self.dropout
        ])

        self.block_2 = tf.keras.Sequential([
            self.conv2d_1,
            self.conv1d_1,
            self.conv1d_2,
            self.avg_pool,
            self.dropout
        ])


        # Global average pooling, flatten, and dense layer
        self.global_avg_pool = GlobalAveragePooling2D(keepdims=False)
        self.flatten = Flatten()
        self.dense = Dense(units=10, activation='softmax')  # 10 classes for MNIST

    def call(self, inputs):
        x = self.block_1(inputs)
        x = self.block_2(x)
        x = self.global_avg_pool(x)
        x = self.flatten(x)
        x = self.dense(x)
        return x


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal((1, 28, 28, 1))  # Dummy input tensor (batch_size, height, width, channels)
    output_tensor = model(input_tensor)
    print("Output shape:", output_tensor.shape)