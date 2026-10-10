import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Add, MaxPool2D, Flatten, Concatenate, Dense
from tensorflow.keras.models import Model

class DL_Model(Model):
    def __init__(self):
        super(DL_Model, self).__init__()
        # First block: main path and branch path
        self.conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')
        self.conv2 = Conv2D(filters=1, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')
        self.add = Add()

        # Second block: multi-scale pooling
        self.pool1 = MaxPool2D(pool_size=(1, 1), strides=(1, 1), padding='valid')
        self.pool2 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid')
        self.pool3 = MaxPool2D(pool_size=(4, 4), strides=(4, 4), padding='valid')
        self.flatten = Flatten()
        self.concatenate = Concatenate(axis=-1)

        # Fully connected layers
        self.dense1 = Dense(units=128, activation='relu')
        self.dense2 = Dense(units=10, activation='softmax')

    def call(self, inputs):
        # First block processing
        main_path = self.conv1(inputs)
        main_path = self.conv2(main_path)
        branch_path = inputs
        block1_output = self.add([main_path, branch_path])

        # Second block processing
        pool1_output = self.pool1(block1_output)
        pool2_output = self.pool2(block1_output)
        pool3_output = self.pool3(block1_output)

        # Flatten and concatenate
        flat1 = self.flatten(pool1_output)
        flat2 = self.flatten(pool2_output)
        flat3 = self.flatten(pool3_output)
        block2_output = self.concatenate([flat1, flat2, flat3])

        # Fully connected layers
        fc_output = self.dense1(block2_output)
        output = self.dense2(fc_output)

        return output

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal((32, 28, 28, 1))  # Batch size of 32, 28x28 grayscale images
    output_tensor = model(input_tensor)
    print(output_tensor.shape)