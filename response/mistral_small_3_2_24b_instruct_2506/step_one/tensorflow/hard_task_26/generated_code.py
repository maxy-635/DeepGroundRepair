import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPool2D, UpSampling2D, Dense, Concatenate
from tensorflow.keras.models import Model

class DL_Model(Model):
    def __init__(self):
        super(DL_Model, self).__init__()
        # Main path components
        self.initial_conv = Conv2D(filters=64, kernel_size=(1, 1), activation='relu')
        self.branch1_conv = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')
        self.branch2_pool = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')
        self.branch2_conv = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')
        self.branch2_upsample = UpSampling2D(size=(2, 2))
        self.branch3_pool = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')
        self.branch3_conv = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')
        self.branch3_upsample = UpSampling2D(size=(2, 2))
        self.concat_conv = Conv2D(filters=64, kernel_size=(1, 1), activation='relu')

        # Branch path components
        self.branch_path_conv = Conv2D(filters=64, kernel_size=(1, 1), activation='relu')

        # Classification layers
        self.flatten = tf.keras.layers.Flatten()
        self.fc1 = Dense(units=128, activation='relu')
        self.fc2 = Dense(units=10, activation='softmax')

    def call(self, inputs):
        # Main path
        x = self.initial_conv(inputs)
        branch1 = self.branch1_conv(x)

        branch2 = self.branch2_pool(x)
        branch2 = self.branch2_conv(branch2)
        branch2 = self.branch2_upsample(branch2)

        branch3 = self.branch3_pool(x)
        branch3 = self.branch3_conv(branch3)
        branch3 = self.branch3_upsample(branch3)

        concatenated = tf.experimental.numpy.concatenate([branch1, branch2, branch3], axis=-1)
        main_output = self.concat_conv(concatenated)

        # Branch path
        branch_output = self.branch_path_conv(inputs)

        # Combine main and branch paths
        combined = tf.math.add(main_output, branch_output)

        # Classification
        x = self.flatten(combined)
        x = self.fc1(x)
        output = self.fc2(x)

        return output

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal(shape=(1, 32, 32, 3))  # CIFAR-10 input shape
    output_tensor = model(input_tensor)
    print("Model output shape:", output_tensor.shape)