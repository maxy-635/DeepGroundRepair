import tensorflow as tf
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Reshape, Multiply, Conv2D, Add, Flatten
from tensorflow.keras.models import Model

class DL_Model(Model):
    def __init__(self):
        super(DL_Model, self).__init__()
        # Main path layers
        self.global_avg_pool = GlobalAveragePooling2D(keepdims='mask')
        self.fc1 = Dense(units=128, activation='relu')
        self.fc2 = Dense(units='mask', activation='relu')
        self.reshape = Reshape(target_shape=(-1, 1, 1))
        self.multiply = Multiply()

        # Branch path layers
        self.conv2d = Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')

        # Final layers
        self.flatten = Flatten()
        self.fc3 = Dense(units=256, activation='relu')
        self.fc4 = Dense(units=128, activation='relu')
        self.fc5 = Dense(units=10, activation='softmax')

    def call(self, inputs):
        # the initial shape of inputs is: (1, 32, 32, 3)
        # Main path
        x = self.global_avg_pool(inputs)  # output tensor shape of this layer should be (1, 32, 32, 3)
        x = self.fc1(x)
        x = self.fc2(x)  # output tensor shape of this layer should be (1, 1, 1, 3)
        x = self.reshape(x)
        x = self.multiply([inputs, x])

        # Branch path
        branch = self.conv2d(inputs)

        # Combine paths
        combined = Add()([x, branch])

        # Final layers
        combined = self.flatten(combined)
        combined = self.fc3(combined)
        combined = self.fc4(combined)
        output = self.fc5(combined)

        return output

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal(shape=(1, 32, 32, 3))  # CIFAR-10 input shape
    output_tensor = model(input_tensor)
    print("Model output shape:", output_tensor.shape)