import tensorflow as tf
from tensorflow.keras.layers import Conv2D, GlobalAveragePooling2D, Dense, Add
from tensorflow.keras.models import Model

class DL_Model(Model):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Main path
        self.global_avg_pool = GlobalAveragePooling2D(keepdims=False)
        self.dense1 = Dense(units=1024, activation='relu')
        self.dense2 = Dense(units=32, activation='relu')

        # Branch path
        self.conv2d = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')

        # Fully connected layers for final classification
        self.dense3 = Dense(units=1024, activation='relu')
        self.dense4 = Dense(units=10, activation='relu')
        self.dense5 = Dense(units=10, activation='softmax')

    def call(self, inputs):
        # the initial shape of inputs is: (1, 32, 32, 3)
        # Main path
        main_path = self.global_avg_pool(inputs)  # output tensor shape: (1, 3)
        main_path = self.dense1(main_path)  # output tensor shape: (1, 1024)
        main_path = self.dense2(main_path)  # output tensor shape: (1, 32)

        # Reshape main path output to match input shape
        main_path = tf.reshape(main_path, shape=(1, 1024))

        # Branch path
        branch_path = self.conv2d(inputs)

        # Add the outputs of the main and branch paths
        merged_path = Add()([main_path, branch_path])

        # Fully connected layers for classification
        x = self.dense3(merged_path)
        x = self.dense4(x)
        output = self.dense5(x)

        return output


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal(shape=(1, 32, 32, 3))  # CIFAR-10 input shape
    output_tensor = model(input_tensor)
    print(output_tensor.shape)