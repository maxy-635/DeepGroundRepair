import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input

class DL_Model(Model):
    def __init__(self):
        super(DL_Model, self).__init__()
        # First block
        self.max_pool_1x1 = tf.raw_ops.MaxPoolV2(input='mask', ksize='mask', strides='mask', padding='mask')
        self.max_pool_2x2 = tf.raw_ops.MaxPoolV2(input=None, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        self.max_pool_4x4 = tf.raw_ops.MaxPoolV2(input=None, ksize=[1, 4, 4, 1], strides=[1, 4, 4, 1], padding='SAME')
        self.flatten_1 = Flatten()
        self.flatten_2 = Flatten()
        self.flatten_3 = Flatten()
        self.dropout = tf.nn.dropout
        self.dense_1 = Dense(units=128, activation='relu')
        self.reshape = tf.raw_ops.Reshape(tensor=None, shape=[-1, 4, 4, 8])

        # Second block
        self.separable_conv_1x1 = tf.nn.separable_conv2d(input=None, depthwise_filter=None, pointwise_filter=None, strides=[1, 1, 1, 1], padding='SAME')
        self.separable_conv_3x3 = tf.nn.separable_conv2d(input=None, depthwise_filter=None, pointwise_filter=None, strides=[1, 1, 1, 1], padding='SAME')
        self.separable_conv_5x5 = tf.nn.separable_conv2d(input=None, depthwise_filter=None, pointwise_filter=None, strides=[1, 1, 1, 1], padding='SAME')
        self.separable_conv_7x7 = tf.nn.separable_conv2d(input=None, depthwise_filter=None, pointwise_filter=None, strides=[1, 1, 1, 1], padding='SAME')
        self.concatenate = tf.experimental.numpy.concatenate
        self.flatten_4 = Flatten()
        self.dense_2 = Dense(units=10, activation='softmax')

    def call(self, inputs):
        # First block
        pool_1x1 = self.max_pool_1x1(input=inputs)
        pool_2x2 = self.max_pool_2x2(input=inputs)
        pool_4x4 = self.max_pool_4x4(input=inputs)

        flat_1x1 = self.flatten_1(pool_1x1)
        flat_2x2 = self.flatten_2(pool_2x2)
        flat_4x4 = self.flatten_3(pool_4x4)

        dropout_1x1 = self.dropout(flat_1x1, rate=0.5)
        dropout_2x2 = self.dropout(flat_2x2, rate=0.5)
        dropout_4x4 = self.dropout(flat_4x4, rate=0.5)

        concat = tf.concat([dropout_1x1, dropout_2x2, dropout_4x4], axis=1)
        dense_out = self.dense_1(concat)
        reshaped = self.reshape(tensor=dense_out)

        # Second block
        # Split the input into four groups along the last dimension
        split = tf.split(reshaped, num_or_size_splits=4, axis=-1)

        # Process each group with separable convolution
        conv_1x1 = self.separable_conv_1x1(input=split[0], depthwise_filter=tf.constant(0.1, shape=[1, 1, 8, 1]), pointwise_filter=tf.constant(0.1, shape=[1, 1, 8, 8]))
        conv_3x3 = self.separable_conv_3x3(input=split[1], depthwise_filter=tf.constant(0.1, shape=[3, 3, 8, 1]), pointwise_filter=tf.constant(0.1, shape=[1, 1, 8, 8]))
        conv_5x5 = self.separable_conv_5x5(input=split[2], depthwise_filter=tf.constant(0.1, shape=[5, 5, 8, 1]), pointwise_filter=tf.constant(0.1, shape=[1, 1, 8, 8]))
        conv_7x7 = self.separable_conv_7x7(input=split[3], depthwise_filter=tf.constant(0.1, shape=[7, 7, 8, 1]), pointwise_filter=tf.constant(0.1, shape=[1, 1, 8, 8]))

        # Concatenate the outputs
        concat_conv = self.concatenate([conv_1x1, conv_3x3, conv_5x5, conv_7x7], axis=-1)

        # Final classification
        flat_final = self.flatten_4(concat_conv)
        output = self.dense_2(flat_final)

        return output

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.keras.Input(shape=(32, 32, 3))
    output_tensor = model(input_tensor)
    print(output_tensor)