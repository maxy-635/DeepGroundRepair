import tensorflow as tf
from tensorflow.keras.layers import Conv2D, AveragePooling2D, Concatenate, Flatten, Dense

class DL_Model(tf.keras.Model):
    def __init__(self):
        super(DL_Model, self).__init__()
        self.conv2d_1 = Conv2D(filters=1, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')
        self.conv2d_2 = Conv2D(filters=1, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')
        self.conv2d_3 = Conv2D(filters=1, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')
        self.avg_pool_1 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')
        self.avg_pool_2 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')
        self.avg_pool_3 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')
        self.concat = Concatenate(axis=-1)
        self.flatten = Flatten()
        self.dense_1 = Dense(units=128, activation='relu')
        self.dense_2 = Dense(units=10, activation='softmax')

    def call(self, inputs):
        group_1 = inputs[:, ::3, :, :]
        group_2 = inputs[:, 1::3, :, :]
        group_3 = inputs[:, 2::3, :, :]

        out_1 = self.conv2d_1(group_1)
        out_2 = self.conv2d_2(group_2)
        out_3 = self.conv2d_3(group_3)

        pool_1 = self.avg_pool_1(out_1)
        pool_2 = self.avg_pool_2(out_2)
        pool_3 = self.avg_pool_3(out_3)

        concatenated = self.concat([pool_1, pool_2, pool_3])
        flattened = self.flatten(concatenated)
        dense_1_out = self.dense_1(flattened)
        output = self.dense_2(dense_1_out)

        return output


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal((1, 32, 32, 3))  # Example input tensor with shape (batch_size, height, width, channels)
    output_tensor = model(input_tensor)
    print(output_tensor.shape)