import tensorflow as tf
from tensorflow.keras.layers import Conv2D, GlobalAveragePooling2D, Dense, Reshape, Multiply, Add, Flatten
from tensorflow.keras.models import Model

class DL_Model(Model):
    def __init__(self, input_shape=(32, 32, 3), num_classes=10):
        super(DL_Model, self).__init__()
        self.input_shape = input_shape
        self.num_classes = num_classes

        # Main path
        self.conv1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')
        self.conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')
        self.global_avg_pool = GlobalAveragePooling2D(keepdims=True)
        self.dense1 = Dense(units=64, activation='relu')
        self.dense2 = Dense(units=3 * 32 * 32, activation='sigmoid')  # Assuming input_shape is (32, 32, 3)
        self.reshape = Reshape(target_shape=(32, 32, 3))

        # Branch path
        self.branch = tf.keras.layers.Lambda(lambda x: x)

        # Final layers
        self.flatten = Flatten()
        self.final_dense1 = Dense(units=128, activation='relu')
        self.final_dense2 = Dense(units=num_classes, activation='softmax')

    def call(self, inputs):
        # Main path
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.global_avg_pool(x)
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.reshape(x)

        # Branch path
        branch = self.branch(inputs)

        # Combine main and branch paths
        combined = Add()([x, branch])

        # Final layers
        x = self.flatten(combined)
        x = self.final_dense1(x)
        x = self.final_dense2(x)

        return x

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal(shape=(1, 32, 32, 3))
    output_tensor = model(input_tensor)
    print("Output shape:", output_tensor.shape)