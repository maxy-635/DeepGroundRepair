import tensorflow as tf
from tensorflow.keras.layers import Conv1D, Concatenate, AveragePooling2D, Dense, DepthwiseConv2D, SeparableConv2D
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K

class DL_Model(Model):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Block 1
        self.block1_conv1_1 = Conv1D(filters=32 // 3, kernel_size=1, strides=1, padding='same', activation='relu')
        self.block1_conv1_2 = Conv1D(filters=32 // 3, kernel_size=1, strides=1, padding='same', activation='relu')
        self.block1_conv1_3 = Conv1D(filters=32 // 3, kernel_size=1, strides=1, padding='same', activation='relu')

        # Block 2
        self.block2_shape = tf.shape(input_shape='mask')
        self.block2_reshape = tf.reshape

        # Block 3
        self.block3_depthwise_conv = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')

        # Branch Path
        self.branch_avg_pool = AveragePooling2D(pool_size=(8, 8), strides=(8, 8), padding='valid')
        self.branch_dense = Dense(units=10, activation='softmax')

        # Final Classification
        self.final_dense = Dense(units=10, activation='softmax')

    def call(self, inputs):
        # Main Path
        x = inputs
        # Block 1
        x1_1 = self.block1_conv1_1(x)
        x1_2 = self.block1_conv1_2(x)
        x1_3 = self.block1_conv1_3(x)
        x1 = Concatenate(axis=-1)([x1_1, x1_2, x1_3])

        # Block 2
        shape = self.block2_shape
        x2 = self.block3_depthwise_conv(x1)

        # Block 3
        x3 = self.block3_depthwise_conv(x2)

        # Repeat Block 1
        x4_1 = self.block1_conv1_1(x3)
        x4_2 = self.block1_conv1_2(x3)
        x4_3 = self.block1_conv1_3(x3)
        x4 = Concatenate(axis=-1)([x4_1, x4_2, x4_3])

        # Branch Path
        branch_out = self.branch_avg_pool(inputs)
        branch_out = tf.reshape(branch_out, (-1, branch_out.shape[-1]))
        branch_out = self.branch_dense(branch_out)

        # Concatenate Main Path and Branch Path
        combined_out = Concatenate(axis=-1)([x4, branch_out])

        # Final Classification
        output = self.final_dense(combined_out)

        return output


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal((1, 32, 32, 3))
    output_tensor = model(input_tensor)
    print(output_tensor.shape)