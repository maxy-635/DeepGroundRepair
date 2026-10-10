import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPool2D, Concatenate, GlobalAveragePooling2D, Dense, Lambda, Multiply, Reshape, Flatten, Activation
from tensorflow.keras.models import Model

class DL_Model(Model):
    def __init__(self):
        super(DL_Model, self).__init__()
        # Block 1: Parallel branches
        self.conv1x1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')
        self.conv3x3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')
        self.conv5x5 = Conv2D(filters='mask', kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')
        self.maxpool3x3 = MaxPool2D(pool_size='mask', strides='mask', padding='mask')

        # Concatenate the outputs of the parallel branches
        self.concat = Concatenate(axis=-1)

        # Block 2: Global average pooling and fully connected layers
        self.global_avg_pool = GlobalAveragePooling2D(keepdims=True)

        # Define the custom weight generation layer
        self.fc1 = Dense(units=64*64, activation='relu')  # Assuming input channels are 64*64
        self.fc2 = Dense(units=64*64, activation='relu')  # Same as input channels
        self.reshape = Reshape(target_shape=(64, 64, 1))  # Reshape to match input feature map shape

        # Multiply the input feature map with the generated weights
        self.multiply = Multiply()

    def call(self, inputs):
        # the initial shape of inputs is: (1, 32, 32, 3)
        # Block 1: Parallel branches
        conv1x1_out = self.conv1x1(inputs)  # output tensor shape: (1, 32, 32, 64)
        conv3x3_out = self.conv3x3(inputs)  # output tensor shape: (1, 32, 32, 64)
        conv5x5_out = self.conv5x5(inputs)  # output tensor shape of this layer should be (1, 32, 32, 1)
        maxpool_out = self.maxpool3x3(inputs)  # output tensor shape of this layer should be (1, 32, 32, 3)

        # Concatenate the outputs
        concat_out = self.concat([conv1x1_out, conv3x3_out, conv5x5_out, maxpool_out])

        # Block 2: Global average pooling and fully connected layers
        pooled_features = self.global_avg_pool(concat_out)  # output tensor shape: (1, 1, 1, 195)

        # Generate weights
        weights = self.fc1(pooled_features)  # output tensor shape: (1, 1, 1, 4096)
        weights = self.fc2(weights)  # output tensor shape: (1, 1, 1, 4096)
        weights = self.reshape(weights)  # output tensor shape: (1, 64, 64, 1)

        # Multiply the input feature map with the generated weights
        weighted_features = self.multiply([concat_out, weights])

        # Final fully connected layer for classification
        flattened = Flatten()(weighted_features)
        output = Dense(units=10, activation='softmax')(flattened)

        return output

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal(shape=(1, 32, 32, 3))  # CIFAR-10 image shape
    output_tensor = model(input_tensor)
    print("Output shape:", output_tensor.shape)