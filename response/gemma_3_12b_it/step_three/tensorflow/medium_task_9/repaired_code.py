import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, ReLU, AveragePooling2D, Flatten, Dense

class DL_Model(tf.keras.Model):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Basic block
        self.conv1_main = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same', activation=None)
        self.bn1_main = BatchNormalization(axis=-1)
        self.relu1_main = ReLU(negative_slope=0.01)

        self.conv1_branch = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same', activation=None)
        self.bn1_branch = BatchNormalization(axis=-1)
        self.relu1_branch = ReLU(negative_slope=0.01)

        self.conv2_main = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same', activation=None)
        self.bn2_main = BatchNormalization(axis=-1)
        self.relu2_main = ReLU(negative_slope=0.01)

        self.conv2_branch = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same', activation=None)
        self.bn2_branch = BatchNormalization(axis=-1)
        self.relu2_branch = ReLU(negative_slope=0.01)

        self.avg_pool = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')
        self.flatten = Flatten()
        self.dense = Dense(units=10, activation='softmax')  # Assuming 10 classes for CIFAR-10

    def call(self, inputs):
        # the initial shape of inputs is: (1, 32, 32, 3)
        # Main path
        x = self.conv1_main(inputs)  # output tensor shape: (1, 32, 32, 16)
        x = self.bn1_main(x)  # output tensor shape: (1, 32, 32, 16)
        x = self.relu1_main(x)  # output tensor shape: (1, 32, 32, 16)

        # Branch path
        branch = self.conv1_branch(inputs)  # output tensor shape: (1, 32, 32, 16)
        branch = self.bn1_branch(branch)  # output tensor shape: (1, 32, 32, 16)
        branch = self.relu1_branch(branch)  # output tensor shape: (1, 32, 32, 16)

        # Feature fusion (addition)
        x = tf.keras.backend.update_add(x, branch)

        # Main path
        x = self.conv2_main(x)
        x = self.bn2_main(x)
        x = self.relu2_main(x)

        # Branch path
        branch = self.conv2_branch(x)
        branch = self.bn2_branch(branch)
        branch = self.relu2_branch(branch)

        # Feature fusion (addition)
        x = tf.keras.backend.update_add(x, branch)

        x = self.avg_pool(x)
        x = self.flatten(x)
        x = self.dense(x)
        return x


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal((1, 32, 32, 3))  # Dummy input (batch_size=1, height=32, width=32, channels=3)
    output_tensor = model(input_tensor)
    print(output_tensor.shape)