import tensorflow as tf
from tensorflow.keras.layers import Conv2D, SeparableConv2D, ReLU, Concatenate, Add, Flatten, Dense, Lambda, Reshape
from tensorflow.keras.models import Model

class DL_Model(Model):
    def __init__(self):
        super(DL_Model, self).__init__()
        # Main path components
        self.separable_conv1 = SeparableConv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same')
        self.relu1 = ReLU()
        self.separable_conv2 = SeparableConv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same')
        self.relu2 = ReLU()
        self.separable_conv3 = SeparableConv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same')
        self.relu3 = ReLU()

        # Branch path component
        self.branch_conv = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same')

        # Feature fusion
        self.add = Add()

        # Classification head
        self.flatten = Flatten()
        self.dense = Dense(units=10, activation='softmax')

    def call(self, inputs):
        # Main path processing
        x = inputs
        for i in range(3):
            # Get current input
            current_input = x

            # Process through ReLU and separable conv
            x = self.relu1(x)
            x = self.separable_conv1(x)

            # Concatenate with original input
            x = Concatenate(axis=-1)([current_input, x])

        # Branch path processing
        branch_output = self.branch_conv(inputs)

        # Feature fusion
        fused_features = self.add([x, branch_output])

        # Classification
        output = self.flatten(fused_features)
        output = self.dense(output)

        return output

if __name__ == "__main__":
    # Create model instance
    model = DL_Model()

    # Create dummy input tensor (batch_size=1, height=28, width=28, channels=1)
    input_tensor = tf.random.normal(shape=(1, 28, 28, 1))

    # Get model output
    output_tensor = model(input_tensor)

    print("Input shape:", input_tensor.shape)
    print("Output shape:", output_tensor.shape)