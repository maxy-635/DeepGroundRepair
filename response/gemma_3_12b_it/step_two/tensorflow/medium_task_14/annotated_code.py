import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, ReLU, Add, Dense
from tensorflow.keras.models import Model, Sequential

class DL_Model(Model):
    def __init__(self):
        super(DL_Model, self).__init__()
        # Define the three sequential blocks
        self.block1 = Sequential([
            Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'),
            BatchNormalization(axis=-1),
            ReLU()
        ])
        self.block2 = Sequential([
            Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'),
            BatchNormalization(axis=-1),
            ReLU()
        ])
        self.block3 = Sequential([
            Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'),
            BatchNormalization(axis=-1),
            ReLU()
        ])

        # Define the parallel branch
        self.parallel_branch = Sequential([
            Conv2D(filters='mask', kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'),
            BatchNormalization(axis=-1),
            ReLU()
        ])

        # Fully connected layers
        self.dense1 = Dense(units=128, activation='relu')
        self.dense2 = Dense(units=10, activation='softmax')  # 10 classes for CIFAR-10

    def call(self, inputs):
        # Process the input through the three sequential blocks
        output_block1 = self.block1(inputs)
        output_block2 = self.block2(output_block1)
        output_block3 = self.block3(output_block2)

        # Process the input through the parallel branch
        output_parallel = self.parallel_branch(inputs)  # output tensor shape of this layer should be (32, 32, 256)

        # Add the outputs from all paths
        added_output = Add()([output_block3, output_parallel])

        # Pass the aggregated result through the fully connected layers
        flattened_output = tf.keras.layers.Flatten()(added_output)
        dense1_output = self.dense1(flattened_output)
        output = self.dense2(dense1_output)

        return output


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 with 3 channels
    output_tensor = model(input_tensor)
    print(output_tensor)