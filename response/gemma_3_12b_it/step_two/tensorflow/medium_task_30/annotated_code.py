import tensorflow as tf
from tensorflow.keras.layers import AveragePooling2D, Flatten, Concatenate, Dense

class DL_Model(tf.keras.Model):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Average Pooling Layers
        self.avg_pool1 = AveragePooling2D(pool_size=(1, 1), strides=(1, 1), padding='valid')
        self.avg_pool2 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')
        self.avg_pool3 = AveragePooling2D(pool_size=(4, 4), strides=(4, 4), padding='valid')

        # Flatten Layers
        self.flatten1 = Flatten()
        self.flatten2 = Flatten()

        # Concatenate Layer
        self.concatenate = Concatenate(axis=-1)

        # Fully Connected Layers
        self.dense1 = Dense(units=128, activation='relu')
        self.dense2 = Dense(units=10, activation='softmax')  # Assuming 10 classes for CIFAR-10

    def call(self, inputs):
        # Pass through Average Pooling Layers
        x = self.avg_pool1(inputs)
        x = self.avg_pool2(x)
        x = self.avg_pool3(x)

        # Flatten and Concatenate
        x = self.flatten1(x)
        x = self.flatten2(x)
        x = self.concatenate([x, self.flatten2(self.avg_pool1(inputs)), self.flatten2(self.avg_pool2(inputs)), self.flatten2(self.avg_pool3(inputs))])

        # Pass through Fully Connected Layers
        x = self.dense1(x)
        output = self.dense2(x)

        return output


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal((1, 32, 32, 3))  # Example input shape for CIFAR-10 (batch_size, height, width, channels)
    output_tensor = model(input_tensor)
    print(output_tensor.shape)