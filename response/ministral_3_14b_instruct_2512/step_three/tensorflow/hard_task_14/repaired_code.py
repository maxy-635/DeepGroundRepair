import tensorflow as tf
from tensorflow.keras.layers import Conv2D, GlobalAveragePooling2D, Dense, Reshape, Multiply, Add, Flatten
from tensorflow.keras.models import Model

class DL_Model(Model):
    def __init__(self, input_shape=(32, 32, 3), num_classes=10):
        super(DL_Model, self).__init__()
        self.input_shapes = input_shape
        self.num_classes = num_classes

        # Main path
        self.global_avg_pool = GlobalAveragePooling2D(keepdims=True)
        self.dense1 = Dense(units=128, activation='relu')
        self.dense2 = Dense(units=input_shape[-1], activation='sigmoid')  # Output channels same as input channels
        self.reshape = Reshape(target_shape=(1, 1, 3))

        # Branch path
        self.conv_branch = Conv2D(filters=input_shape[-1], kernel_size=(3, 3), strides=(1, 1), padding='same')

        # Final layers
        self.flatten = Flatten()
        self.fc1 = Dense(units=128, activation='relu')
        self.fc2 = Dense(units=128, activation='relu')
        self.fc3 = Dense(units=num_classes, activation='softmax')

    def call(self, inputs):
        # the initial shape of inputs is: (1, 32, 32, 3)
        # Main path
        x = self.global_avg_pool(inputs)  # output tensor shape: (1, 1, 1, 3)
        x = self.dense1(x)  # output tensor shape: (1, 1, 1, 128)
        x = self.dense2(x)  # output tensor shape: (1, 1, 1, 3)
        x = self.reshape(x)
        x = Multiply()([inputs, x])

        # Branch path
        branch = self.conv_branch(inputs)

        # Combine paths
        combined = Add()([x, branch])

        # Final layers
        x = self.flatten(combined)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)

        return x

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal(shape=(1, 32, 32, 3))  # Batch size of 1, 32x32 RGB image
    output_tensor = model(input_tensor)
    print("Output shape:", output_tensor.shape)