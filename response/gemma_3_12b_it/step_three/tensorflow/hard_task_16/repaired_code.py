import tensorflow as tf
from tensorflow.keras.layers import Conv2D, DepthwiseConv2D, BatchNormalization, ReLU, Concatenate, GlobalMaxPool2D, Dense, Reshape, Add

class DL_Model(tf.keras.Model):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Block 1
        self.conv1_1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu', name='conv1_1')
        self.conv1_2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', name='conv1_2')
        self.conv1_3 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu', name='conv1_3')

        # Transition Convolution
        self.transition_conv = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu', name='transition_conv')

        # Block 2
        self.global_max_pool = GlobalMaxPool2D(name='global_max_pool')
        self.dense1 = Dense(units=64, activation='relu', name='dense1')
        self.dense2 = Dense(units=64, activation='relu', name='dense2')
        self.reshape = Reshape(target_shape=(64, 1), name='reshape')
        self.multiply = tf.keras.layers.Multiply(name='multiply')

        # Branch
        self.branch_conv = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu', name='branch_conv')

        # Classification
        self.add = Add(name='add')
        self.dense_out = Dense(units=10, activation='softmax', name='dense_out')

    def call(self, inputs):
        # Main Path
        x = tf.split(inputs, num_or_size_splits=3, axis=-1)
        x1 = self.conv1_1(x[0])
        x2 = self.conv1_2(x[1])
        x3 = self.conv1_3(x[2])
        x = Concatenate(axis=-1)([x1, x2, x3])
        x = self.transition_conv(x)

        x = self.global_max_pool(x)
        weights = self.dense1(x)
        weights = self.dense2(weights)
        weights = self.reshape(weights)
        x = self.multiply([x, weights])

        # Branch
        branch = self.branch_conv(inputs)

        # Add and Classify
        merged = self.add([x, branch])
        output = self.dense_out(merged)

        return output


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal(shape=(1, 32, 32, 3))
    output_tensor = model(input_tensor)
    print(output_tensor.shape)