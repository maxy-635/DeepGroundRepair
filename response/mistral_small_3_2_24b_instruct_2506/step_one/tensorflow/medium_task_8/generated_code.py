import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Concatenate, Dense, Flatten, Lambda, Add
from tensorflow.keras.models import Model

class DL_Model(Model):
    def __init__(self):
        super(DL_Model, self).__init__()
        # Main path components
        self.conv2d_1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')
        self.conv2d_2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')
        self.concatenate = Concatenate(axis=-1)
        self.split = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))
        self.combine = Lambda(lambda x: tf.concat(x, axis=-1))

        # Branch path component
        self.branch_conv = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')

        # Fusion and classification
        self.add = Add()
        self.flatten = Flatten()
        self.dense = Dense(units=10, activation='softmax')

    def call(self, inputs):
        # Split the input into three groups
        split_inputs = self.split(inputs)

        # Main path processing
        # First group remains unchanged
        group1 = split_inputs[0]

        # Second group undergoes feature extraction
        group2 = self.conv2d_1(split_inputs[1])

        # Combine second group with third group
        combined = self.combine([group2, split_inputs[2]])
        combined = self.conv2d_2(combined)

        # Concatenate all three groups
        main_output = self.concatenate([group1, combined])

        # Branch path processing
        branch_output = self.branch_conv(inputs)

        # Fusion of main and branch paths
        fused_output = self.add([main_output, branch_output])

        # Classification
        flattened = self.flatten(fused_output)
        output = self.dense(flattened)

        return output

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal(shape=(1, 32, 32, 3))  # CIFAR-10 input shape
    output_tensor = model(input_tensor)
    print("Model output shape:", output_tensor.shape)