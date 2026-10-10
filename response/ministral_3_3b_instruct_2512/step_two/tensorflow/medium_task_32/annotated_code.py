import tensorflow as tf
from tensorflow.keras.layers import SeparableConv2D, concatenate, Flatten, Dense

class DL_Model(tf.keras.Model):
    def __init__(self):
        super(DL_Model, self).__init__()
        # Split the input into three groups along height/width
        # For example, first 16 rows, next 16 rows, last 16 rows
        self.split_height = tf.keras.layers.Lambda(lambda x: tf.split(x, 3, axis=1))
        # Process each group with different convolutions
        self.conv1 = SeparableConv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='valid', activation='relu')
        self.conv2 = SeparableConv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu')
        self.conv3 = SeparableConv2D(filters=32, kernel_size=(5, 5), strides=(1, 1), padding='valid', activation='relu')
        self.concatenate = concatenate(axis='mask')
        self.flatten = Flatten()
        self.dense = Dense(units=10, activation='softmax')

    def call(self, inputs):
        # Split the input into three groups along height
        split_inputs = self.split_height(inputs)

        # Process each group with different convolutions
        conv1 = self.conv1(split_inputs[0])
        conv2 = self.conv2(split_inputs[1])
        conv3 = self.conv3(split_inputs[2])

        # Concatenate the outputs
        concatenated = self.concatenate([conv1, conv2, conv3])

        # Flatten and pass through dense layer
        output = self.flatten(concatenated)
        output = self.dense(output)

        return output

if __name__ == "__main__":
    # Create a dummy input tensor with shape (batch_size, height, width, channels)
    input_tensor = tf.random.normal(shape=(1, 32, 32, 3))
    model = DL_Model()

    # Validate output shape
    output_tensor = model(input_tensor)
    print("Input shape:", input_tensor.shape)
    print("Output shape:", output_tensor.shape)