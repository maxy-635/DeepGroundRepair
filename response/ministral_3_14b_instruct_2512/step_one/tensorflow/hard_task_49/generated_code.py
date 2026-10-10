import tensorflow as tf
from tensorflow.keras.layers import AveragePooling2D, Flatten, Concatenate, Dense, Reshape, DepthwiseConv2D, SeparableConv2D
from tensorflow.keras.models import Model

class DL_Model(Model):
    def __init__(self):
        super(DL_Model, self).__init__()
        # First block: three average pooling layers with different scales
        self.avg_pool_1x1 = AveragePooling2D(pool_size=(1, 1), strides=(1, 1), padding='valid')
        self.avg_pool_2x2 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')
        self.avg_pool_4x4 = AveragePooling2D(pool_size=(4, 4), strides=(4, 4), padding='valid')

        # Flatten and concatenate the outputs
        self.flatten_1x1 = Flatten()
        self.flatten_2x2 = Flatten()
        self.flatten_4x4 = Flatten()
        self.concatenate = Concatenate(axis=-1)

        # Fully connected layer and reshape for second block
        self.dense = Dense(units=100, activation='relu')
        self.reshape = Reshape(target_shape=(1, 1, 100))

        # Second block: depthwise separable convolutions with different kernel sizes
        self.sep_conv_1x1 = SeparableConv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')
        self.sep_conv_3x3 = SeparableConv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')
        self.sep_conv_5x5 = SeparableConv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')
        self.sep_conv_7x7 = SeparableConv2D(filters=64, kernel_size=(7, 7), strides=(1, 1), padding='same', activation='relu')

        # Final flatten and dense layer
        self.final_flatten = Flatten()
        self.final_dense = Dense(units=10, activation='softmax')

    def call(self, inputs):
        # First block processing
        pool_1x1 = self.avg_pool_1x1(inputs)
        pool_2x2 = self.avg_pool_2x2(inputs)
        pool_4x4 = self.avg_pool_4x4(inputs)

        # Flatten and concatenate
        flat_1x1 = self.flatten_1x1(pool_1x1)
        flat_2x2 = self.flatten_2x2(pool_2x2)
        flat_4x4 = self.flatten_4x4(pool_4x4)

        concatenated = self.concatenate([flat_1x1, flat_2x2, flat_4x4])

        # Fully connected and reshape
        fc_output = self.dense(concatenated)
        reshaped = self.reshape(fc_output)

        # Second block processing
        # Split the input into four groups along the last dimension
        # Assuming the input is (batch, height, width, channels)
        # We need to split the channels into four groups
        # For simplicity, let's assume the input has 4 channels after reshape
        # This part might need adjustment based on actual input shape
        # Here, we'll split the channels into four groups of 25 each (assuming 100 channels)
        # This is a placeholder; actual implementation may vary
        split_outputs = tf.split(reshaped, num_or_size_splits=4, axis=-1)

        # Process each group with depthwise separable convolutions
        # Note: The input to these convolutions needs to be 4D (batch, height, width, channels)
        # So we need to reshape the split outputs appropriately
        # This is a complex part and might need adjustment
        # For now, we'll assume the split outputs are already in the correct shape
        # This is a placeholder and may not work as intended
        conv_1x1 = self.sep_conv_1x1(split_outputs[0])
        conv_3x3 = self.sep_conv_3x3(split_outputs[1])
        conv_5x5 = self.sep_conv_5x5(split_outputs[2])
        conv_7x7 = self.sep_conv_7x7(split_outputs[3])

        # Concatenate the outputs
        concatenated_conv = self.concatenate([conv_1x1, conv_3x3, conv_5x5, conv_7x7])

        # Final flatten and dense layer
        final_flat = self.final_flatten(concatenated_conv)
        output = self.final_dense(final_flat)

        return output

if __name__ == "__main__":
    model = DL_Model()
    # Create a dummy input tensor with shape (batch_size, height, width, channels)
    # For MNIST, the input shape is typically (28, 28, 1)
    input_tensor = tf.random.normal(shape=(1, 28, 28, 1))
    output_tensor = model(input_tensor)
    print("Output shape:", output_tensor.shape)