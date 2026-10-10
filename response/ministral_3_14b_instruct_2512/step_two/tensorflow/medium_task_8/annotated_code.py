import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Dense, Flatten, Concatenate, Add, Lambda, BatchNormalization
from tensorflow.keras.models import Model

class DL_Model(Model):
    def __init__(self, num_classes=10):
        super(DL_Model, self).__init__()
        self.num_classes = num_classes

        # Main path components
        self.conv1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')
        self.conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')
        self.bn1 = BatchNormalization()
        self.bn2 = BatchNormalization()

        # Branch path component
        self.conv_branch = Conv2D(filters='mask', kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')

        # Final layers
        self.flatten = Flatten()
        self.dense = Dense(units=num_classes, activation='softmax')

    def call(self, inputs):
        # the initial shape of inputs is: (1, 32, 32, 9)
        # Split input into three groups along the last dimension
        # Assuming input channels are divisible by 3
        input_shape = tf.shape(inputs)  # output tensor shape: (4)
        channels = input_shape[-1]  # output tensor shape: (1)
        split_size = channels // 3  # output tensor shape: (1)

        # Split the input into three parts
        group1 = inputs[:, :, :, :split_size]  # output tensor shape: (1, 32, 32, 3)
        group2 = inputs[:, :, :, split_size:2*split_size]  # output tensor shape: (1, 32, 32, 3)
        group3 = inputs[:, :, :, 2*split_size:]  # output tensor shape: (1, 32, 32, 3)

        # Process second group through conv1
        conv1_out = self.conv1(group2)  # output tensor shape: (1, 32, 32, 64)
        conv1_out = self.bn1(conv1_out)  # output tensor shape: (1, 32, 32, 64)

        # Combine with third group
        combined = tf.concat([conv1_out, group3], axis=-1)  # output tensor shape: (1, 32, 32, 67)

        # Additional conv2
        conv2_out = self.conv2(combined)  # output tensor shape: (1, 32, 32, 64)
        conv2_out = self.bn2(conv2_out)  # output tensor shape: (1, 32, 32, 64)

        # Concatenate all three groups
        main_path_out = tf.concat([group1, conv2_out], axis=-1)  # output tensor shape: (1, 32, 32, 67)

        # Branch path processing
        branch_out = self.conv_branch(inputs)  # output tensor shape of this layer should be (1, 32, 32, 67)

        # Add main and branch paths
        combined_out = Add()([main_path_out, branch_out])

        # Final classification
        flattened = self.flatten(combined_out)
        output = self.dense(flattened)

        return output

if __name__ == "__main__":
    # Create model instance
    model = DL_Model()

    # Create dummy input tensor with shape (batch_size, height, width, channels)
    # CIFAR-10 images are 32x32 with 3 channels, so we'll use 9 channels (divisible by 3)
    input_tensor = tf.random.normal(shape=(1, 32, 32, 9))

    # Get output
    output_tensor = model(input_tensor)
    print("Input shape:", input_tensor.shape)
    print("Output shape:", output_tensor.shape)