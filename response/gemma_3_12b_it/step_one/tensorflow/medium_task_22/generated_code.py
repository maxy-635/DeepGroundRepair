import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPool2D, concatenate, Flatten, Dense

class DL_Model(tf.keras.Model):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Branch 1: 3x3 convolutions
        self.conv2d_3x3_1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', name='conv2d_3x3_1')
        self.conv2d_3x3_2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', name='conv2d_3x3_2')

        # Branch 2: 1x1 followed by two 3x3 convolutions
        self.conv2d_1x1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu', name='conv2d_1x1')
        self.conv2d_3x3_3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', name='conv2d_3x3_3')
        self.conv2d_3x3_4 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', name='conv2d_3x3_4')

        # Branch 3: Max pooling
        self.max_pool = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same', name='max_pool')

        # Feature fusion
        self.flatten = Flatten(name='flatten')
        self.dense_1 = Dense(units=128, activation='relu', name='dense_1')
        self.dense_2 = Dense(units=10, activation='softmax', name='dense_2')  # CIFAR-10 has 10 classes

    def call(self, inputs):
        # Branch 1
        branch_1 = self.conv2d_3x3_1(inputs)
        branch_1 = self.conv2d_3x3_2(branch_1)

        # Branch 2
        branch_2 = self.conv2d_1x1(inputs)
        branch_2 = self.conv2d_3x3_3(branch_2)
        branch_2 = self.conv2d_3x3_4(branch_2)

        # Branch 3
        branch_3 = self.max_pool(inputs)

        # Concatenate the branches
        merged_features = concatenate([branch_1, branch_2, branch_3], axis=-1)

        # Flatten and fully connected layers
        x = self.flatten(merged_features)
        x = self.dense_1(x)
        output = self.dense_2(x)

        return output


if __name__ == "__main__":
    model = DL_Model()
    # Dummy input tensor for CIFAR-10 (32x32x3)
    input_tensor = tf.random.normal((1, 32, 32, 3))  # Batch size of 1
    output_tensor = model(input_tensor)
    print("Output shape:", output_tensor.shape)