import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Add, concatenate

class DL_Model(tf.keras.Model):
    def __init__(self):
        super(DL_Model, self).__init__()
        # Main path
        self.main_path = tf.keras.Sequential([
            Conv2D(32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'),
            Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'),
            MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid')
        ])

        # Branch path
        self.branch_path = tf.keras.Sequential([
            Conv2D(64, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')
        ])

        # Combine paths
        self.combine = Add()

        # Fully connected layers
        self.flatten = Flatten()
        self.fc1 = Dense(128, activation='relu')
        self.fc2 = Dense(10, activation='relu')

    def call(self, inputs):
        main_output = self.main_path(inputs)
        branch_output = self.branch_path(inputs)

        # Combine outputs using concatenate
        combined_output = concatenate([main_output, branch_output], axis=-1)
        # Use Add instead of concatenate for element-wise addition
        # combined_output = self.combine([main_output, branch_output])

        flattened_output = self.flatten(combined_output)
        output = self.fc1(flattened_output)
        output = self.fc2(output)

        return output

if __name__ == "__main__":
    # Create model instance
    model = DL_Model()

    # Create dummy input tensor with shape (batch_size, height, width, channels)
    # For CIFAR-10, height=32, width=32, channels=3
    input_tensor = tf.random.normal((1, 32, 32, 3))
    output_tensor = model(input_tensor)
    print("Input shape:", input_tensor.shape)
    print("Output shape:", output_tensor.shape)