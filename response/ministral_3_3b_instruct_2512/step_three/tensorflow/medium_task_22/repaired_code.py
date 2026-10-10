import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPool2D, Concatenate, Flatten, Dense, Dropout, BatchNormalization

class DL_Model(tf.keras.Model):
    def __init__(self):
        super(DL_Model, self).__init__()
        # Branch 1: 3x3 convolutions
        self.branch1_conv1 = Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')
        self.branch1_conv2 = Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')

        # Branch 2: 1x1 convolution followed by two 3x3 convolutions
        self.branch2_conv1 = Conv2D(64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')
        self.branch2_conv3 = Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')
        self.branch2_conv4 = Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')

        # Branch 3: Max pooling
        self.branch3_pool = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')

        # Multi-scale feature fusion block
        self.concatenate = Concatenate(axis=1)

        # Fully connected layers
        self.flatten = Flatten()
        self.fc1 = Dense(128, activation='relu')
        self.fc2 = Dense(10, activation='softmax')

    def call(self, inputs):
        # Branch 1 processing
        branch1_output = self.branch1_conv1(inputs)
        branch1_output = self.branch1_conv2(branch1_output)

        # Branch 2 processing
        branch2_output = self.branch2_conv1(inputs)
        branch2_output = self.branch2_conv3(branch2_output)
        branch2_output = self.branch2_conv4(branch2_output)

        # Branch 3 processing
        branch3_output = self.branch3_pool(inputs)

        # Multi-scale feature fusion
        fused_features = self.concatenate([branch1_output, branch2_output, branch3_output])

        # Fully connected layers
        x = self.flatten(fused_features)
        x = self.fc1(x)
        x = self.fc2(x)

        return x

if __name__ == "__main__":
    # Create a dummy input tensor with shape (batch_size, height, width, channels)
    input_tensor = tf.random.normal(shape=(1, 32, 32, 3))
    model = DL_Model()
    output_tensor = model(input_tensor)
    print("Input shape:", input_tensor.shape)
    print("Output shape:", output_tensor.shape)