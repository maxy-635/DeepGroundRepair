import tensorflow as tf
from tensorflow.keras.layers import AveragePooling2D, Conv2D, Flatten, Dense, Dropout, Activation

class DL_Model(tf.keras.Model):
    def __init__(self):
        super(DL_Model, self).__init__()
        self.avg_pool = AveragePooling2D(pool_size=(5, 5), strides=(3, 3), padding='valid')
        self.conv_1x1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')
        self.flatten = Flatten()
        self.dense1 = Dense(units=128, activation='relu')
        self.dropout = Dropout(rate=0.5)
        self.dense2 = Dense(units=10, activation='softmax')

    def call(self, inputs):
        x = self.avg_pool(inputs)
        x = self.conv_1x1(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dropout(x)
        x = self.dense2(x)
        return x

if __name__ == "__main__":
    # Dummy input tensor with shape (batch_size, height, width, channels)
    # For MNIST, height=28, width=28, channels=1
    input_tensor = tf.random.normal((1, 28, 28, 1))
    output_tensor = DL_Model()(input_tensor)
    print("Input shape:", input_tensor.shape)
    print("Output shape:", output_tensor.shape)