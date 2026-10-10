import tensorflow as tf
from tensorflow.keras import layers

class DL_Model(tf.keras.Model):
    def __init__(self):
        super(DL_Model, self).__init__()
        self.conv3x3 = layers.Conv2D(64, kernel_size=(3, 3), activation='relu')
        self.conv5x5 = layers.Conv2D(64, kernel_size=(5, 5), activation='relu')
        self.pool = layers.MaxPool2D(pool_size=(2, 2))
        self.avg_pool = layers.GlobalAveragePooling2D()
        self.fc1 = layers.Dense(128, activation='relu')
        self.fc2 = layers.Dense(10)

    def call(self, inputs):
        x = self.conv3x3(inputs)
        y = self.conv5x5(inputs)
        x = self.pool(x)
        y = self.pool(y)
        x = self.avg_pool(x)
        y = self.avg_pool(y)
        x = self.fc1(x)
        y = self.fc1(y)
        return self.fc2(x + y)

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal((1, 32, 32, 3))
    output_tensor = model(input_tensor)
    print(output_tensor.shape)