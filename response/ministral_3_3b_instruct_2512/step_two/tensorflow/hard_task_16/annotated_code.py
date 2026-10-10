import tensorflow as tf
from tensorflow.keras import layers

class DL_Model(tf.keras.Model):
    def __init__(self):
        super(DL_Model, self).__init__()
        # Block 1: Split input into three groups and concatenate outputs
        self.conv1_1x1 = layers.Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same')
        self.conv2_3x3 = layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same')
        self.conv3_1x1 = layers.Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same')

        # Transition Convolution
        self.trans_conv = layers.Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same')

        # Block 2: Global max pooling and weight generation
        self.global_pool = layers.GlobalMaxPool2D(keepdims=True)
        self.fc1 = layers.Dense(units=64, activation='relu')
        self.fc2 = layers.Dense(units=64, activation='relu')
        self.reshape = layers.Reshape(target_shape=(-1, 64))

        # Main path
        self.main_path = tf.keras.Sequential([
            self.conv1_1x1,
            self.conv2_3x3,
            self.conv3_1x1,
            self.trans_conv,
            self.global_pool,
            self.fc1,
            self.fc2,
        ])

        # Branch (direct connection to input)
        self.branch = layers.Add()

        # Final classification layer
        self.classifier = layers.Dense(10, activation='softmax')

    def call(self, inputs):
        # the initial shape of inputs is: (1, 32, 32, 3)
        # Main path
        main_output = self.main_path(inputs)  # output tensor shape: (1, 1, 1, 64)

        # Branch (direct connection to input)
        branch_output = inputs  # output tensor shape: (1, 32, 32, 3)

        # Add outputs
        combined = self.branch([main_output, branch_output])

        # Reshape for final classification
        reshaped = self.reshape(combined)

        # Classification
        return self.classifier(reshaped)

if __name__ == "__main__":
    # Create model instance
    model = DL_Model()

    # Dummy input tensor for validation (CIFAR-10 images are 32x32 with 3 channels)
    input_tensor = tf.random.normal(shape=(1, 32, 32, 3))

    # Forward pass
    output_tensor = model(input_tensor)
    print("Input shape:", input_tensor.shape)
    print("Output shape:", output_tensor.shape)
    print("Output values:", output_tensor.numpy())