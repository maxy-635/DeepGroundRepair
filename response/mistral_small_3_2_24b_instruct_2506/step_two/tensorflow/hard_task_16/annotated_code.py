import tensorflow as tf
from tensorflow.keras.layers import Dense, GlobalMaxPool2D
from tensorflow.keras.models import Model

class DL_Model(Model):
    def __init__(self):
        super(DL_Model, self).__init__()
        # Block 1 components
        self.conv1x1_1 = tf.keras.layers.Conv2D(filters=64, kernel_size=(1, 1), padding='same')
        self.conv3x3_1 = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), padding='same')
        self.conv1x1_2 = tf.keras.layers.Conv2D(filters=64, kernel_size=(1, 1), padding='same')
        self.conv1x1_3 = tf.keras.layers.Conv2D(filters=64, kernel_size=(1, 1), padding='same')
        self.conv3x3_2 = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), padding='same')
        self.conv1x1_4 = tf.keras.layers.Conv2D(filters=64, kernel_size=(1, 1), padding='same')
        self.conv1x1_5 = tf.keras.layers.Conv2D(filters=64, kernel_size=(1, 1), padding='same')
        self.conv3x3_3 = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), padding='same')
        self.conv1x1_6 = tf.keras.layers.Conv2D(filters=64, kernel_size=(1, 1), padding='same')

        # Transition convolution
        self.trans_conv = tf.keras.layers.Conv2D(filters=64, kernel_size=(1, 1), padding='same')

        # Block 2 components
        self.global_max_pool = GlobalMaxPool2D(keepdims=False)
        self.dense1 = Dense(units=64, activation='relu')
        self.dense2 = Dense(units='mask', activation='sigmoid')

        # Final classification layer
        self.classification = Dense(units=10, activation='softmax')

    def call(self, inputs):
        # the initial shape of inputs is: (1, 32, 32, 3)
        # Block 1: Split into three groups and process
        split1 = self.conv1x1_1(inputs)  # output tensor shape: (1, 32, 32, 64)
        split1 = self.conv3x3_1(split1)  # output tensor shape: (1, 32, 32, 64)
        split1 = self.conv1x1_2(split1)  # output tensor shape: (1, 32, 32, 64)

        split2 = self.conv1x1_3(inputs)  # output tensor shape: (1, 32, 32, 64)
        split2 = self.conv3x3_2(split2)  # output tensor shape: (1, 32, 32, 64)
        split2 = self.conv1x1_4(split2)  # output tensor shape: (1, 32, 32, 64)

        split3 = self.conv1x1_5(inputs)  # output tensor shape: (1, 32, 32, 64)
        split3 = self.conv3x3_3(split3)  # output tensor shape: (1, 32, 32, 64)
        split3 = self.conv1x1_6(split3)  # output tensor shape: (1, 32, 32, 64)

        # Concatenate the splits
        concatenated = tf.experimental.numpy.concatenate([split1, split2, split3], axis=-1)  # output tensor shape: (1, 32, 32, 192)

        # Transition convolution
        adjusted_output = self.trans_conv(concatenated)  # output tensor shape: (1, 32, 32, 64)

        # Block 2: Global max pooling and weight generation
        pooled = self.global_max_pool(adjusted_output)  # output tensor shape: (1, 64)
        weights = self.dense1(pooled)  # output tensor shape: (1, 64)
        weights = self.dense2(weights)  # output tensor shape of this layer should be (1, 3)

        # Reshape and multiply
        reshaped_weights = tf.raw_ops.Reshape(tensor=weights, shape=[-1, 1, 1, 64])
        main_path_output = tf.math.multiply(adjusted_output, reshaped_weights)

        # Branch connection
        branch_output = inputs  # output tensor shape: (1, 32, 32, 3)
        combined_output = tf.math.add(main_path_output, branch_output)

        # Final classification
        output = self.classification(combined_output)

        return output

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal(shape=(1, 32, 32, 3))  # CIFAR-10 input shape
    output_tensor = model(input_tensor)
    print("Input shape:", input_tensor.shape)
    print("Output shape:", output_tensor.shape)