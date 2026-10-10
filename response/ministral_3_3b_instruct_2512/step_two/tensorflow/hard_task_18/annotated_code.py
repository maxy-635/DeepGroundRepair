import tensorflow as tf
from tensorflow.keras.layers import Conv2D, AveragePooling2D, Add, GlobalAveragePooling2D, Dense, Reshape, Flatten

class DL_Model:
    def __init__(self):
        self.conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same')
        self.pool1 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')
        self.conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same')
        self.pool2 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')
        self.add_layer = Add()
        self.gap = GlobalAveragePooling2D(keepdims=True)
        self.dense1 = Dense(units=64, activation='relu')
        self.dense2 = Dense(units=32, activation='relu')
        self.reshape = Reshape(target_shape=(64, 32))
        self.flatten = Flatten()
        self.output_layer = Dense(units=10, activation='softmax')

    def __call__(self, input_tensor):
        x = self.conv1(input_tensor)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        main_path = x
        skip_connection = x
        x = self.add_layer([x, main_path])
        x = self.gap(x)
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.reshape(x)
        x = tf.multiply(x, skip_connection)
        x = self.flatten(x)
        x = self.output_layer(x)
        return x

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal(shape=(1, 32, 32, 3))
    output_tensor = model(input_tensor)
    print(output_tensor.shape)