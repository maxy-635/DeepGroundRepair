import tensorflow as tf
from tensorflow.keras.layers import MaxPool2D, Flatten, Concatenate, Dense, Reshape, Conv2D, Lambda
from tensorflow.keras.models import Sequential

class DL_Model(tf.keras.Model):
    def __init__(self):
        super(DL_Model, self).__init__()
        self.block1 = Sequential([
            MaxPool2D(pool_size=(1, 1), strides=(1, 1), padding='valid'),
            MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid'),
            MaxPool2D(pool_size=(4, 4), strides=(4, 4), padding='valid'),
            Flatten(),
            Concatenate(),
            Dense(units=1024, activation='relu'),
            Reshape(target_shape=(4, 128))
        ])

        self.block2 = Sequential([
            Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='valid', activation='relu'),
            Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu'),
            Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='valid', activation='relu'),
            MaxPool2D(pool_size=(3, 3), strides=(3, 3), padding='valid'),
            Concatenate(),
            Flatten(),
            Dense(units=10, activation='softmax')
        ])

    def call(self, inputs):
        block1_output = self.block1(inputs)
        block2_output = self.block2(block1_output)
        return block2_output

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal(shape=(1, 28, 28, 1))
    output_tensor = model(input_tensor)
    print(output_tensor.shape)