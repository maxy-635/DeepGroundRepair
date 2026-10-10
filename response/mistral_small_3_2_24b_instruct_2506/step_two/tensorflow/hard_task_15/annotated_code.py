import tensorflow as tf
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Reshape, Multiply, Add
from tensorflow.keras.models import Model

class DL_Model(Model):
    def __init__(self):
        super(DL_Model, self).__init__()
        # Main path layers
        self.global_avg_pool = GlobalAveragePooling2D(keepdims=True)
        self.fc1 = Dense(units=64, activation='relu')
        self.fc2 = Dense(units=64, activation='sigmoid')
        self.reshape = Reshape(target_shape=(-1, 1, 1, 64))
        self.multiply = Multiply()
        # Branch path (identity connection)
        self.add = Add()
        # Final layers
        self.fc3 = Dense(units=64, activation='relu')
        self.fc4 = Dense(units=10, activation='softmax')

    def call(self, inputs):
        # the initial shape of inputs is: (2, 32, 32, 3)
        # Main path
        x = self.global_avg_pool(inputs)  # output tensor shape: (2, 1, 1, 3)
        x = self.fc1(x)  # output tensor shape: (2, 1, 1, 64)
        x = self.fc2(x)  # output tensor shape: (2, 1, 1, 64)
        x = self.reshape(x)  # output tensor shape: (2, 1, 1, 1, 64)
        x = self.multiply([inputs, x])

        # Branch path
        branch = inputs

        # Combine paths
        combined = self.add([x, branch])

        # Final layers
        combined = self.fc3(combined)
        combined = self.fc4(combined)

        return combined

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal(shape=(2, 32, 32, 3))  # CIFAR-10 input shape
    output_tensor = model(input_tensor)
    print(output_tensor)