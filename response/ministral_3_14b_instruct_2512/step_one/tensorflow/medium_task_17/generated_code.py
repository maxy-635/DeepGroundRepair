import tensorflow as tf
from tensorflow.keras.layers import Reshape, Permute, Dense, Activation
from tensorflow.keras.models import Model

class DL_Model(Model):
    def __init__(self, input_shape, num_classes):
        super(DL_Model, self).__init__()
        self.input_shapes = input_shape
        self.num_classes = num_classes
        self.reshape1 = Reshape(target_shape=(input_shape[0], input_shape[1], 3, input_shape[2]//3))
        self.permute = Permute(dims=(0, 1, 3, 2))
        self.reshape2 = Reshape(target_shape=input_shape)
        self.dense = Dense(units=num_classes, activation='softmax')

    def call(self, inputs):
        x = self.reshape1(inputs)
        x = self.permute(x)
        x = self.reshape2(x)
        x = self.dense(x)
        return x

if __name__ == "__main__":
    model = DL_Model(input_shape=(32, 32, 3), num_classes=10)
    input_tensor = tf.random.normal(shape=(None, 32, 32, 3))
    output_tensor = model(input_tensor)
    print(output_tensor.shape)