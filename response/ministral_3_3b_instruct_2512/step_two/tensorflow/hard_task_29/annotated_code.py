import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Add, MaxPool2D, Flatten, Dense, concatenate

class DL_Model(tf.keras.Model):
    def __init__(self):
        super(DL_Model, self).__init__()
        # First block: main path and branch path
        self.conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')
        self.conv2 = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')
        self.branch_path = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='same', activation=None)

        # Second block: three max pooling layers
        self.pool1 = MaxPool2D(pool_size=(1, 1), strides=(1, 1), padding='valid')
        self.pool2 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid')
        self.pool3 = MaxPool2D(pool_size=(4, 4), strides=(4, 4), padding='valid')

        # Fully connected layers
        self.flatten = Flatten()
        self.fc1 = Dense(units=128, activation='relu')
        self.fc2 = Dense(units=10, activation='softmax')

    def call(self, inputs):
        # First block: main path and branch path
        main_path = self.conv1(inputs)
        main_path = self.conv2(main_path)
        branch_path = self.branch_path(inputs)
        combined = Add()([main_path, branch_path])

        # Second block: three max pooling layers
        pool1_output = self.pool1(combined)
        pool2_output = self.pool2(combined)
        pool3_output = self.pool3(combined)

        # Flatten and concatenate outputs from pooling layers
        flattened_pool1 = self.flatten(pool1_output)
        flattened_pool2 = self.flatten(pool2_output)
        flattened_pool3 = self.flatten(pool3_output)

        concatenated_output = concatenate([flattened_pool1, flattened_pool2, flattened_pool3])

        # Fully connected layers
        output = self.fc1(concatenated_output)
        output = self.fc2(output)

        return output

if __name__ == "__main__":
    # Create a dummy input tensor with shape (batch_size, height, width, channels)
    input_tensor = tf.random.normal(shape=(1, 28, 28, 1))
    model = DL_Model()
    output_tensor = model(input_tensor)
    print("Input shape:", input_tensor.shape)
    print("Output shape:", output_tensor.shape)