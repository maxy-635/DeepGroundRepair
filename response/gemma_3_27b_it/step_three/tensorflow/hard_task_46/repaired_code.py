import tensorflow as tf

class DL_Model(tf.keras.Model):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Block 1: Separable Convolution with different kernel sizes
        self.sep_conv1 = tf.keras.layers.SeparableConv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')
        self.sep_conv2 = tf.keras.layers.SeparableConv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')
        self.sep_conv3 = tf.keras.layers.SeparableConv2D(filters=32, kernel_size=(5, 5), padding='same', activation='relu')
        self.concat1 = tf.keras.layers.Concatenate(axis=-1)

        # Block 2: Multiple Branches for Enhanced Feature Extraction
        self.conv_branch = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')
        self.conv1_branch = tf.keras.layers.Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')
        self.conv2_branch = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')
        self.conv3_branch = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')
        self.maxpool_branch = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(1, 1), padding='same')
        self.concat2 = tf.keras.layers.Concatenate(axis=-1)

        # Final Layers
        self.global_avg_pool = tf.keras.layers.GlobalAveragePooling2D(keepdims=False)
        self.dense = tf.keras.layers.Dense(units=10, activation='softmax')

    def call(self, inputs):
        # the initial shape of inputs is: (1, 32, 32, 3)
        # Block 1
        x1 = self.sep_conv1(inputs)  # output tensor shape: (1, 32, 32, 32)
        x2 = self.sep_conv2(inputs)  # output tensor shape: (1, 32, 32, 32)
        x3 = self.sep_conv3(inputs)  # output tensor shape: (1, 32, 32, 32)
        x = self.concat1([x1, x2, x3])  # output tensor shape: (1, 32, 32, 96)

        # Block 2
        branch1 = self.conv_branch(x)  # output tensor shape: (1, 32, 32, 32)
        branch2 = self.conv1_branch(x)  # output tensor shape: (1, 32, 32, 32)
        branch2 = self.conv2_branch(branch2)  # output tensor shape: (1, 32, 32, 32)
        branch2 = self.conv3_branch(branch2)  # output tensor shape: (1, 32, 32, 32)
        branch3 = self.maxpool_branch(x)  # output tensor shape of this layer should be (1, 32, 32, 96)

        x = self.concat2([branch1, branch2, branch3])

        # Final Layers
        x = self.global_avg_pool(x)
        x = self.dense(x)
        return x


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal(shape=(1, 32, 32, 3))  # CIFAR-10 input shape
    output_tensor = model(input_tensor)
    print(output_tensor.shape)