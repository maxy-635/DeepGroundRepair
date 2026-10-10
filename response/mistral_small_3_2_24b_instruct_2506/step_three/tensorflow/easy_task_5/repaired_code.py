import tensorflow as tf
from tensorflow.keras import layers

class DL_Model(tf.keras.Model):
    def __init__(self):
        super(DL_Model, self).__init__()
        self.conv1x1_reduce = layers.Conv2D(filters=32, kernel_size=(1, 1), activation='relu')
        self.conv3x3_feature = layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu')
        self.conv1x1_restore = layers.Conv2D(filters=32, kernel_size=(1, 1), activation='relu')
        self.flatten = layers.Flatten()
        self.dense = layers.Dense(units=10, activation='softmax')

    def call(self, inputs):
        x = self.conv1x1_reduce(inputs)
        x = self.conv3x3_feature(x)
        x = self.conv1x1_restore(x)
        x = self.flatten(x)
        return self.dense(x)

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal(shape=(1, 28, 28, 1))  # MNIST image shape: (28, 28, 1)
    output_tensor = model(input_tensor)
    print("Input shape:", input_tensor.shape)
    print("Output shape:", output_tensor.shape)