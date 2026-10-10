import tensorflow as tf
from tensorflow.keras.layers import Conv2D, AveragePooling2D, Conv2DTranspose, Concatenate, Add, Dense, Flatten, Lambda
from tensorflow.keras.models import Model

class DL_Model(Model):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Main path
        self.conv1x1_main = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')

        # Branch 1
        self.conv3x3_branch1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')

        # Branch 2
        self.avg_pool_branch2 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')
        self.conv3x3_branch2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')
        self.conv2dtranspose_branch2 = Conv2DTranspose(filters=64, kernel_size=(2, 2), strides=(2, 2), padding='same', activation='relu')

        # Branch 3
        self.avg_pool_branch3 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')
        self.conv3x3_branch3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')
        self.conv2dtranspose_branch3 = Conv2DTranspose(filters=64, kernel_size=(2, 2), strides=(2, 2), padding='same', activation='relu')

        # Concatenate and 1x1 conv
        self.concat = Concatenate(axis=-1)
        self.conv1x1_after_concat = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')

        # Branch path
        self.conv1x1_branch_path = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')

        # Add and final layers
        self.add = Add()
        self.flatten = Flatten()
        self.dense = Dense(units=10, activation='softmax')

    def call(self, inputs):
        # Main path
        x_main = self.conv1x1_main(inputs)

        # Branch 1
        branch1 = self.conv3x3_branch1(x_main)

        # Branch 2
        branch2 = self.avg_pool_branch2(x_main)
        branch2 = self.conv3x3_branch2(branch2)
        branch2 = self.conv2dtranspose_branch2(branch2)

        # Branch 3
        branch3 = self.avg_pool_branch3(x_main)
        branch3 = self.conv3x3_branch3(branch3)
        branch3 = self.conv2dtranspose_branch3(branch3)

        # Concatenate and 1x1 conv
        concat_output = self.concat([branch1, branch2, branch3])
        concat_output = self.conv1x1_after_concat(concat_output)

        # Branch path
        branch_path = self.conv1x1_branch_path(inputs)

        # Add main and branch paths
        add_output = self.add([concat_output, branch_path])

        # Final layers
        flattened = self.flatten(add_output)
        output = self.dense(flattened)

        return output

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal(shape=(1, 32, 32, 3))  # CIFAR-10 image shape
    output_tensor = model(input_tensor)
    print(output_tensor.shape)