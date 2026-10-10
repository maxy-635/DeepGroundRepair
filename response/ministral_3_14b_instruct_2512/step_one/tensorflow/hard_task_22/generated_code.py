import tensorflow as tf
from tensorflow.keras.layers import Conv2D, SeparableConv2D, Concatenate, Add, Flatten, Dense, Lambda, Activation, BatchNormalization, MaxPool2D
from tensorflow.keras.models import Model

class DL_Model(Model):
    def __init__(self):
        super(DL_Model, self).__init__()
        # Main path - three groups with different kernel sizes
        self.sep_conv1x1 = SeparableConv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')
        self.sep_conv3x3 = SeparableConv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')
        self.sep_conv5x5 = SeparableConv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')

        # Branch path - 1x1 convolution
        self.branch_conv = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')

        # Concatenate and add layers
        self.concat = Concatenate(axis=-1)
        self.add = Add()

        # Pooling and flattening
        self.max_pool = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')
        self.flatten = Flatten()

        # Fully connected layers
        self.dense1 = Dense(units=128, activation='relu')
        self.dense2 = Dense(units=10, activation='softmax')

    def call(self, inputs):
        # Main path
        # Split input into three groups along channel dimension
        input_shape = tf.shape(inputs)
        channels = input_shape[-1]
        split_size = channels // 3

        # Split the input into three parts along the channel axis
        group1 = inputs[:, :, :, :split_size]
        group2 = inputs[:, :, :, split_size:2*split_size]
        group3 = inputs[:, :, :, 2*split_size:]

        # Apply separable convolutions to each group
        group1_out = self.sep_conv1x1(group1)
        group2_out = self.sep_conv3x3(group2)
        group3_out = self.sep_conv5x5(group3)

        # Concatenate the outputs
        main_path_out = self.concat([group1_out, group2_out, group3_out])

        # Branch path
        branch_path_out = self.branch_conv(inputs)

        # Fuse the outputs
        fused_out = self.add([main_path_out, branch_path_out])

        # Apply pooling
        pooled_out = self.max_pool(fused_out)

        # Flatten and pass through dense layers
        flattened = self.flatten(pooled_out)
        dense_out1 = self.dense1(flattened)
        output = self.dense2(dense_out1)

        return output

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal(shape=(32, 32, 32, 3))  # Batch size 32, height 32, width 32, channels 3
    output_tensor = model(input_tensor)
    print("Output shape:", output_tensor.shape)