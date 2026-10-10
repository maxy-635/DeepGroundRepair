import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, Add, AveragePooling2D, Flatten, Dense

class DL_Model(tf.keras.Model):
    def __init__(self):
        super(DL_Model, self).__init__()
        # Initial convolutional layer
        self.conv1 = Conv2D(filters=16, kernel_size=(3, 3), padding='same')

        # First basic block
        self.conv2 = Conv2D(filters=16, kernel_size=(3, 3), padding='same')
        self.bn2 = BatchNormalization(axis=-1)
        self.relu2 = tf.nn.relu

        # Second basic block
        self.conv3 = Conv2D(filters=16, kernel_size=(3, 3), padding='same')
        self.bn3 = BatchNormalization(axis=-1)
        self.relu3 = tf.nn.relu

        # Branch convolutional layer
        self.conv_branch = Conv2D(filters=16, kernel_size=(1, 1), padding='same')

        # Final layers
        self.avg_pool = AveragePooling2D(pool_size=(8, 8), strides=(1, 1), padding='valid')
        self.flatten = Flatten()
        self.dense = Dense(units=10, activation='softmax')

    def call(self, inputs):
        # Initial convolution
        x = self.conv1(inputs)

        # First basic block
        main_path = self.conv2(x)
        main_path = self.bn2(main_path)
        main_path = self.relu2(main_path)
        branch = x
        out = tf.math.add(main_path, branch)

        # Second basic block
        main_path = self.conv3(out)
        main_path = self.bn3(main_path)
        main_path = self.relu3(main_path)
        branch = self.conv_branch(out)
        out = tf.math.add(main_path, branch)

        # Final layers
        out = self.avg_pool(out)
        out = self.flatten(out)
        out = self.dense(out)

        return out

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.keras.Input(shape=(32, 32, 3))
    output_tensor = model(input_tensor)
    print(output_tensor)