import tensorflow as tf
from tensorflow.keras.layers import Conv1D, SeparableConv2D, Concatenate, Add, Flatten, Dense, BatchNormalization, GlobalAveragePooling2D, MaxPool2D

class DL_Model(tf.keras.Model):
    def __init__(self):
        super(DL_Model, self).__init__()
        self.conv1 = Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=(32, 32, 3))
        self.conv2 = Conv1D(filters=64, kernel_size=3, activation='relu')
        self.flatten = Flatten()
        self.dense = Dense(10)  # Example output layer for classification

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.flatten(x)
        return self.dense(x)

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal((1, 32, 32, 3))  # Example input shape for CIFAR-10 (batch_size=1, height=32, width=32, channels=3)
    output_tensor = model(input_tensor)
    print(output_tensor.shape)