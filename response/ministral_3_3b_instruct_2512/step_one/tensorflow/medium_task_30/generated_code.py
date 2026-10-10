import tensorflow as tf
from tensorflow.keras.layers import AveragePooling2D, Flatten, Dense, Concatenate

class DL_Model(tf.keras.Model):
    def __init__(self):
        super(DL_Model, self).__init__()
        self.pool1 = AveragePooling2D(pool_size=(1, 2), strides=(1, 2), padding='valid')
        self.pool2 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')
        self.pool3 = AveragePooling2D(pool_size=(4, 4), strides=(4, 4), padding='valid')
        self.flatten1 = Flatten()
        self.flatten2 = Flatten()
        self.flatten3 = Flatten()
        self.concatenate = Concatenate(axis=-1)
        self.fc1 = Dense(units=128, activation='relu')
        self.fc2 = Dense(units=10, activation='softmax')

    def call(self, inputs):
        x = inputs
        x1 = self.pool1(x)
        x2 = self.pool2(x1)
        x3 = self.pool3(x2)
        x1_flat = self.flatten1(x1)
        x2_flat = self.flatten2(x2)
        x3_flat = self.flatten3(x3)
        concatenated = self.concatenate([x1_flat, x2_flat, x3_flat])
        x = self.fc1(concatenated)
        output = self.fc2(x)
        return output

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal(shape=(32, 32, 32, 3))  # Dummy input with shape (batch_size, height, width, channels)
    output_tensor = model(input_tensor)
    print(output_tensor.shape)  # Expected output shape: (32, 10)