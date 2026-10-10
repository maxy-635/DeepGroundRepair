import tensorflow as tf
from tensorflow.keras.layers import Lambda, Conv2D, DepthwiseConv2D, SeparableConv2D, Concatenate, Flatten, Dense
from tensorflow.keras.models import Model

class DL_Model(Model):
    def __init__(self, num_classes=10):
        super(DL_Model, self).__init__()
        self.num_classes = num_classes

        # Split the input into three groups along the last dimension (channels)
        self.split = Lambda(lambda x: tf.split(x, 3, axis=-1))

        # Feature extraction branches with different kernel sizes
        self.branch1 = SeparableConv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')
        self.branch2 = SeparableConv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')
        self.branch3 = SeparableConv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')

        # Concatenate the outputs
        self.concat = Concatenate(axis=-1)

        # Classification head
        self.flatten = Flatten()
        self.dense = Dense(units=num_classes, activation='softmax')

    def call(self, inputs):
        # Split the input into three groups
        split_outputs = self.split(inputs)

        # Process each branch
        branch1_out = self.branch1(split_outputs[0])
        branch2_out = self.branch2(split_outputs[1])
        branch3_out = self.branch3(split_outputs[2])

        # Concatenate the outputs
        concatenated = self.concat([branch1_out, branch2_out, branch3_out])

        # Flatten and classify
        flattened = self.flatten(concatenated)
        output = self.dense(flattened)

        return output

if __name__ == "__main__":
    # Create model instance
    model = DL_Model()

    # Create dummy input tensor with shape (batch_size, height, width, channels)
    # CIFAR-10 images are 32x32 with 3 channels
    input_tensor = tf.random.normal(shape=(1, 32, 32, 3))

    # Forward pass
    output_tensor = model(input_tensor)

    print("Input shape:", input_tensor.shape)
    print("Output shape:", output_tensor.shape)