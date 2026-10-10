import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPool2D, UpSampling2D, Concatenate, Dense, add

class DL_Model(tf.keras.Model):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Main Path
        self.conv2d_initial = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')

        # Branch 1
        self.conv2d_branch1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')

        # Branch 2
        self.maxpool2d_branch2 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')
        self.conv2d_branch2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')
        self.upsampling2d_branch2 = UpSampling2D(size=(2, 2))

        # Branch 3
        self.maxpool2d_branch3 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')
        self.conv2d_branch3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')
        self.upsampling2d_branch3 = UpSampling2D(size=(2, 2))

        self.conv2d_concat = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')

        # Branch Path
        self.conv2d_branch_path = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')

        self.dense1 = Dense(units=128, activation='relu')
        self.dense2 = Dense(units=10, activation='softmax')

    def call(self, inputs):
        # Main Path
        x = self.conv2d_initial(inputs)

        # Branch 1
        x1 = self.conv2d_branch1(x)

        # Branch 2
        x2 = self.maxpool2d_branch2(x)
        x2 = self.conv2d_branch2(x2)
        x2 = self.upsampling2d_branch2(x2)

        # Branch 3
        x3 = self.maxpool2d_branch3(x)
        x3 = self.conv2d_branch3(x3)
        x3 = self.upsampling2d_branch3(x3)

        # Concatenate and final conv2d
        x = Concatenate()([x1, x2, x3])
        x = self.conv2d_concat(x)

        # Branch Path
        branch_path = self.conv2d_branch_path(inputs)

        # Add paths
        x = add([x, branch_path])

        # Fully connected layers
        x = self.dense1(x)
        output = self.dense2(x)

        return output


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal((1, 32, 32, 3))  # Dummy input (batch_size, height, width, channels)
    output_tensor = model(input_tensor)
    print(output_tensor.shape)