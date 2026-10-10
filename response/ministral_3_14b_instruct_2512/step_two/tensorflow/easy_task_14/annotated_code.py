import tensorflow as tf
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Reshape, Multiply, Flatten
from tensorflow.keras.models import Model

class DL_Model(Model):
    def __init__(self, num_classes=10):
        super(DL_Model, self).__init__()
        self.global_avg_pool = GlobalAveragePooling2D(keepdims='mask')
        self.dense1 = Dense(units=64, activation='relu')
        self.dense2 = Dense(units=32, activation='relu')
        self.dense3 = Dense(units='mask', activation='relu')
        self.reshape = Reshape(target_shape=(3, 1, 1))  # Reshape to match input channels
        self.multiply = Multiply()
        self.flatten = Flatten()
        self.dense_output = Dense(units=num_classes, activation='softmax')

    def call(self, inputs):
        # the initial shape of inputs is: (32, 32, 32, 3)
        # Compress input features with global average pooling
        pooled = self.global_avg_pool(inputs)  # output tensor shape of this layer should be (32, 32, 32, 3)

        # Generate weights using two fully connected layers
        weights = self.dense1(pooled)
        weights = self.dense2(weights)
        weights = self.dense3(weights)  # output tensor shape of this layer should be (32, 1, 1, 3)

        # Reshape weights to match input channels and multiply
        weights = self.reshape(weights)
        output = self.multiply([inputs, weights])

        # Flatten and produce final output
        output = self.flatten(output)
        output = self.dense_output(output)

        return output

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal(shape=(32, 32, 32, 3))  # Batch of 32, 32x32x3 input
    output_tensor = model(input_tensor)
    print(output_tensor.shape)