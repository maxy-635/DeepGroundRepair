import tensorflow as tf
from tensorflow.keras.layers import Conv2D, AveragePooling2D, Flatten, Dense, Add, LeakyReLU, BatchNormalization, Dropout

class DL_Model(tf.keras.Model):
    def __init__(self):
        super(DL_Model, self).__init__()
        self.path1 = tf.keras.Sequential([
            Conv2D(32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'),
            Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'),
            AveragePooling2D(pool_size=(1, 1), strides=(1, 1), padding='same')
        ])
        self.path2 = tf.keras.Sequential([
            Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')
        ])
        self.flatten = Flatten()
        self.fc = Dense(10, activation='softmax')

    def call(self, inputs):
        # the initial shape of inputs is: (1, 32, 32, 3)
        path1_output = self.path1(inputs)  # output tensor shape of this layer should be (1, 32, 32, 64)
        path2_output = self.path2(inputs)  # output tensor shape: (1, 32, 32, 64)
        combined_output = Add()([path1_output, path2_output])
        output = self.flatten(combined_output)
        output = self.fc(output)
        return output

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal(shape=(1, 32, 32, 3))
    output_tensor = model(input_tensor)
    print(output_tensor.shape)