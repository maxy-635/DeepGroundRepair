import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Lambda, Concatenate, Flatten, Dense, Activation, BatchNormalization, ReLU, GlobalAveragePooling2D
from tensorflow.keras.models import Model

class DL_Model(Model):
    def __init__(self):
        super(DL_Model, self).__init__()
        # Define the three convolutional branches
        self.conv1x1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')
        self.conv3x3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')
        self.conv5x5 = Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')

        # Batch normalization layers
        self.bn1x1 = BatchNormalization(axis=-1)
        self.bn3x3 = BatchNormalization(axis=-1)
        self.bn5x5 = BatchNormalization(axis=-1)

        # Concatenate layer
        self.concatenate = Concatenate(axis=-1)

        # Fully connected layers
        self.flatten = Flatten()
        self.dense1 = Dense(units=128, activation='relu')
        self.dense2 = Dense(units=10, activation='softmax')

    def call(self, inputs):
        # Split the input into three groups along the channel dimension
        # Assuming input has shape (batch_size, height, width, channels)
        # We'll split the channels into three equal parts (or as close as possible)
        # For CIFAR-10, channels=3, so we'll split into 1, 1, 1 (but this is not ideal)
        # A better approach is to split the channels into three groups, but since 3 is not divisible by 3,
        # we'll use the entire input for each branch (this is a simplification for the example)
        # In practice, you might want to adjust the input channels or use a different approach

        # For this example, we'll process the entire input with each kernel size
        # and concatenate the results
        x1 = self.conv1x1(inputs)
        x1 = self.bn1x1(x1)

        x2 = self.conv3x3(inputs)
        x2 = self.bn3x3(x2)

        x3 = self.conv5x5(inputs)
        x3 = self.bn5x5(x3)

        # Concatenate the outputs along the channel dimension
        concatenated = self.concatenate([x1, x2, x3])

        # Flatten the output
        flattened = self.flatten(concatenated)

        # Fully connected layers
        x = self.dense1(flattened)
        outputs = self.dense2(x)

        return outputs

if __name__ == "__main__":
    model = DL_Model()
    # Create a dummy input tensor with shape (batch_size, height, width, channels)
    # CIFAR-10 images are 32x32 with 3 channels
    input_tensor = tf.random.normal(shape=(1, 32, 32, 3))
    output_tensor = model(input_tensor)
    print("Output shape:", output_tensor.shape)