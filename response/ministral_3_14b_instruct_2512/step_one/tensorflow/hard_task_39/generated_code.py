import tensorflow as tf
from tensorflow.keras.layers import Layer, MaxPool2D, Flatten, Concatenate, Dense, Reshape, Conv2D
from tensorflow.keras.models import Model

class DL_Model(Model):
    def __init__(self):
        super(DL_Model, self).__init__()
        # Block 1: Three max pooling layers with different scales
        self.max_pool_1x1 = MaxPool2D(pool_size=(1, 1), strides=(1, 1), padding='valid')
        self.max_pool_2x2 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid')
        self.max_pool_4x4 = MaxPool2D(pool_size=(4, 4), strides=(4, 4), padding='valid')

        # Flatten and concatenate layers
        self.flatten_1 = Flatten()
        self.flatten_2 = Flatten()
        self.flatten_3 = Flatten()
        self.concatenate = Concatenate(axis=-1)

        # Fully connected layer and reshape for Block 2
        self.dense = Dense(units=28*28, activation='relu')  # Assuming MNIST images are 28x28
        self.reshape = Reshape(target_shape=(28, 28, 1))  # Reshape to 4D tensor for Block 2

        # Block 2: Multiple branches
        self.conv1x1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')
        self.conv3x3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')
        self.conv5x5 = Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')
        self.max_pool_3x3 = MaxPool2D(pool_size=(3, 3), strides=(3, 3), padding='valid')

        # Final classification layers
        self.global_avg_pool = tf.keras.layers.GlobalAveragePooling2D()
        self.final_dense = Dense(units=10, activation='softmax')  # 10 classes for MNIST

    def call(self, inputs):
        # Block 1 processing
        pool_1x1 = self.max_pool_1x1(inputs)
        pool_2x2 = self.max_pool_2x2(inputs)
        pool_4x4 = self.max_pool_4x4(inputs)

        # Flatten and concatenate
        flat_1x1 = self.flatten_1(pool_1x1)
        flat_2x2 = self.flatten_2(pool_2x2)
        flat_4x4 = self.flatten_3(pool_4x4)

        # Concatenate the flattened outputs
        concatenated = self.concatenate([flat_1x1, flat_2x2, flat_4x4])

        # Convert to 4D tensor for Block 2
        fc_output = self.dense(concatenated)
        reshaped = self.reshape(fc_output)

        # Block 2 processing (multiple branches)
        branch1 = self.conv1x1(reshaped)
        branch2 = self.conv3x3(reshaped)
        branch3 = self.conv5x5(reshaped)
        branch4 = self.max_pool_3x3(reshaped)

        # Concatenate all branches
        block2_output = self.concatenate([branch1, branch2, branch3, branch4])

        # Final classification
        pooled = self.global_avg_pool(block2_output)
        output = self.final_dense(pooled)

        return output

if __name__ == "__main__":
    # Create model instance
    model = DL_Model()

    # Create dummy input tensor (batch_size=1, height=28, width=28, channels=1)
    input_tensor = tf.random.normal(shape=(1, 28, 28, 1))

    # Get output tensor
    output_tensor = model(input_tensor)
    print("Input shape:", input_tensor.shape)
    print("Output shape:", output_tensor.shape)