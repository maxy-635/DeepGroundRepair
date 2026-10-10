import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPool2D, GlobalAveragePooling2D, Dense, Reshape, Add

class DL_Model(tf.keras.Model):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Main Path
        self.conv1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')
        self.conv2 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')
        self.conv3 = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')
        self.maxpool = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')

        # Branch Path
        self.gap = GlobalAveragePooling2D(keepdims=False)
        self.fc1 = Dense(units=128, activation='relu')
        self.fc2 = Dense(units=256, activation='relu')
        self.reshape = Reshape((1, 1, 256))

        # Classification Path
        self.fc3 = Dense(units=1024, activation='relu')
        self.fc4 = Dense(units=10, activation='softmax')  # Assuming 10 classes for CIFAR-10

    def call(self, inputs):
        # Main Path
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.maxpool(x)

        # Branch Path
        branch = self.gap(inputs)
        branch = self.fc1(branch)
        branch = self.fc2(branch)
        branch = self.reshape(branch)

        # Add the outputs from both paths
        merged = Add()([x, branch])

        # Classification Path
        x = self.fc3(merged)
        output = self.fc4(x)

        return output


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal((1, 32, 32, 3))  # Example input: batch size 1, 32x32 image, 3 channels
    output_tensor = model(input_tensor)
    print("Output shape:", output_tensor.shape)