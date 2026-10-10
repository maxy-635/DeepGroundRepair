import tensorflow as tf
from tensorflow.keras.layers import MaxPool2D, Flatten, Dropout, Dense, Reshape, Conv2D, SeparableConv2D, Concatenate

class DL_Model(tf.keras.Model):
    def __init__(self):
        super(DL_Model, self).__init__()
        # First block: Three max pooling layers with different scales
        self.pool1 = MaxPool2D(pool_size=(1, 1), strides=(1, 1), padding='valid')
        self.pool2 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid')
        self.pool3 = MaxPool2D(pool_size=(4, 4), strides=(4, 4), padding='valid')

        # Flatten each pooling output
        self.flatten1 = Flatten()
        self.flatten2 = Flatten()
        self.flatten3 = Flatten()

        # Concatenate the flattened outputs with dropout
        self.concatenate = Concatenate(axis=-1)
        self.dropout = Dropout(rate=0.5)

        # Fully connected layer and reshape for second block
        self.dense = Dense(units=1024, activation='relu')
        self.reshape = Reshape(target_shape='mask')

        # Second block: Four separable convolutions
        self.sep_conv1 = SeparableConv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')
        self.sep_conv2 = SeparableConv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')
        self.sep_conv3 = SeparableConv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')
        self.sep_conv4 = SeparableConv2D(filters=64, kernel_size=(7, 7), strides=(1, 1), padding='same', activation='relu')

        # Concatenate the outputs of the four separable convolutions
        self.sep_concat = Concatenate(axis=-1)

        # Final classification layer
        self.final_dense = Dense(units=10, activation='softmax')

    def call(self, inputs):
        # the initial shape of inputs is: (1, 32, 32, 3)
        # First block processing
        x = inputs  # output tensor shape: (1, 32, 32, 3)
        x1 = self.pool1(x)  # output tensor shape: (1, 32, 32, 3)
        x1 = self.flatten1(x1)  # output tensor shape: (1, 3072)

        x2 = self.pool2(x)  # output tensor shape: (1, 16, 16, 3)
        x2 = self.flatten2(x2)  # output tensor shape: (1, 768)

        x3 = self.pool3(x)  # output tensor shape: (1, 8, 8, 3)
        x3 = self.flatten3(x3)  # output tensor shape: (1, 192)

        concatenated = self.concatenate([x1, x2, x3])  # output tensor shape: (1, 4032)
        x = self.dropout(concatenated)  # output tensor shape: (1, 4032)

        # Reshape and process through dense layer
        x = self.dense(x)  # output tensor shape: (1, 1024)
        x = self.reshape(x)

        # Second block processing
        x1 = self.sep_conv1(x)
        x2 = self.sep_conv2(x)
        x3 = self.sep_conv3(x)
        x4 = self.sep_conv4(x)

        sep_concat = self.sep_concat([x1, x2, x3, x4])

        # Final classification
        output = self.final_dense(sep_concat)
        return output

if __name__ == "__main__":
    # Create a dummy input tensor with shape (batch_size, height, width, channels)
    # Assuming CIFAR-10 images are 32x32 with 3 channels
    input_tensor = tf.random.normal(shape=(1, 32, 32, 3))
    model = DL_Model()
    output_tensor = model(input_tensor)
    print("Input shape:", input_tensor.shape)
    print("Output shape:", output_tensor.shape)