import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Dense, Add, Concatenate, Flatten, Activation, BatchNormalization, GlobalAveragePooling2D

class DL_Model(tf.keras.Model):
    def __init__(self):
        super(DL_Model, self).__init__()
        # Main path - first feature extraction branch (1x1 conv)
        self.conv1x1_1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')

        # Main path - second feature extraction branch (1x1, 1x7, 7x1 conv sequence)
        self.conv1x1_2 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')
        self.conv1x7_2 = Conv2D(filters=64, kernel_size=(1, 7), strides=(1, 1), padding='same', activation='relu')
        self.conv7x1_2 = Conv2D(filters=64, kernel_size=(7, 1), strides=(1, 1), padding='same', activation='relu')

        # 1x1 conv to align dimensions after concatenation
        self.conv1x1_align = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')

        # Branch connection (identity connection)
        self.branch_conv = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')

        # Classification layers
        self.global_avg_pool = GlobalAveragePooling2D()
        self.dense1 = Dense(units=128, activation='relu')
        self.dense2 = Dense(units=10, activation='softmax')  # 10 classes for CIFAR-10

    def call(self, inputs):
        # Main path - first branch
        x1 = self.conv1x1_1(inputs)

        # Main path - second branch
        x2 = self.conv1x1_2(inputs)
        x2 = self.conv1x7_2(x2)
        x2 = self.conv7x1_2(x2)

        # Concatenate both branches
        concat = Concatenate(axis=-1)([x1, x2])

        # Align dimensions with 1x1 conv
        main_path_output = self.conv1x1_align(concat)

        # Branch connection (identity connection)
        branch_output = self.branch_conv(inputs)

        # Add main path and branch outputs
        add_output = Add()([main_path_output, branch_output])

        # Classification
        x = self.global_avg_pool(add_output)
        x = self.dense1(x)
        x = self.dense2(x)

        return x

if __name__ == "__main__":
    model = DL_Model()
    # CIFAR-10 images are 32x32 with 3 channels
    input_tensor = tf.keras.Input(shape=(32, 32, 3))
    output_tensor = model(input_tensor)
    print("Model created successfully!")
    print("Input shape:", input_tensor.shape)
    print("Output shape:", output_tensor.shape)