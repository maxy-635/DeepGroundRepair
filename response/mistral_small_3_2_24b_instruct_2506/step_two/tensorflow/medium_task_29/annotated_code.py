import tensorflow as tf
from tensorflow.keras.layers import MaxPool2D, Flatten, Concatenate, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input

class DL_Model(Model):
    def __init__(self):
        super(DL_Model, self).__init__()
        # Max pooling layers with different window sizes
        self.max_pool_1x1 = MaxPool2D(pool_size=(1, 1), strides=(1, 1), padding='valid')
        self.max_pool_2x2 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid')
        self.max_pool_4x4 = MaxPool2D(pool_size=(4, 4), strides=(4, 4), padding='valid')

        # Flatten layers
        self.flatten_1x1 = Flatten()
        self.flatten_2x2 = Flatten()
        self.flatten_4x4 = Flatten()

        # Concatenate layer
        self.concatenate = Concatenate(axis=-1)

        # Fully connected layers
        self.dense_1 = Dense(units=128, activation='relu')
        self.dense_2 = Dense(units=10, activation='softmax')

    def call(self, inputs):
        # Apply max pooling and flattening
        x1 = self.flatten_1x1(self.max_pool_1x1(inputs))
        x2 = self.flatten_2x2(self.max_pool_2x2(inputs))
        x3 = self.flatten_4x4(self.max_pool_4x4(inputs))

        # Concatenate the flattened features
        concatenated = self.concatenate([x1, x2, x3])

        # Apply fully connected layers
        x = self.dense_1(concatenated)
        output = self.dense_2(x)

        return output

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = Input(shape=(32, 32, 3))  # CIFAR-10 image shape
    output_tensor = model(input_tensor)
    print(output_tensor)