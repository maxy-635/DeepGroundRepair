import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPool2D, Concatenate, Flatten, Dense, BatchNormalization, Activation, Dropout, Add

class DL_Model(tf.keras.Model):
    def __init__(self):
        super(DL_Model, self).__init__()
        # First convolutional block with 1x1, 3x3, and 5x5 convolutions
        self.conv1x1_1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')
        self.conv3x3_1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')
        self.conv5x5_1 = Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')
        self.maxpool_1 = MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='valid')

        # Second convolutional block with 1x1, 3x3, and 5x5 convolutions
        self.conv1x1_2 = Conv2D(filters=128, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')
        self.conv3x3_2 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')
        self.conv5x5_2 = Conv2D(filters=128, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')
        self.maxpool_2 = MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='valid')

        # Third convolutional block with 1x1, 3x3, and 5x5 convolutions
        self.conv1x1_3 = Conv2D(filters=256, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')
        self.conv3x3_3 = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')
        self.conv5x5_3 = Conv2D(filters=256, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')
        self.maxpool_3 = MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='valid')

        # Fully connected layers
        self.flatten = Flatten()
        self.dense1 = Dense(units=512, activation='relu')
        self.dropout = Dropout(rate=0.5)
        self.dense2 = Dense(units=10, activation='softmax')

    def call(self, inputs):
        # First block
        conv1x1_out_1 = self.conv1x1_1(inputs)
        conv3x3_out_1 = self.conv3x3_1(inputs)
        conv5x5_out_1 = self.conv5x5_1(inputs)
        concat_1 = Concatenate(axis=-1)([conv1x1_out_1, conv3x3_out_1, conv5x5_out_1])
        pool_1 = self.maxpool_1(concat_1)

        # Second block
        conv1x1_out_2 = self.conv1x1_2(pool_1)
        conv3x3_out_2 = self.conv3x3_2(pool_1)
        conv5x5_out_2 = self.conv5x5_2(pool_1)
        concat_2 = Concatenate(axis=-1)([conv1x1_out_2, conv3x3_out_2, conv5x5_out_2])
        pool_2 = self.maxpool_2(concat_2)

        # Third block
        conv1x1_out_3 = self.conv1x1_3(pool_2)
        conv3x3_out_3 = self.conv3x3_3(pool_2)
        conv5x5_out_3 = self.conv5x5_3(pool_2)
        concat_3 = Concatenate(axis=-1)([conv1x1_out_3, conv3x3_out_3, conv5x5_out_3])
        pool_3 = self.maxpool_3(concat_3)

        # Fully connected layers
        flatten_out = self.flatten(pool_3)
        dense1_out = self.dense1(flatten_out)
        dropout_out = self.dropout(dense1_out)
        output = self.dense2(dropout_out)

        return output

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.keras.Input(shape=(32, 32, 3))  # CIFAR-10 image shape
    output_tensor = model(input_tensor)
    print("Model output shape:", output_tensor.shape)