import tensorflow as tf
from tensorflow.keras.layers import AveragePooling2D, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, Concatenate

class DL_Model(Model):
    def __init__(self):
        super(DL_Model, self).__init__()
        # Main path components
        self.conv1x1_main = Conv2D(filters=64, kernel_size=(1, 1), padding='same')
        self.conv3x3_branch1 = Conv2D(filters=64, kernel_size=(3, 3), padding='same')
        self.avg_pool_branch2 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')
        self.conv3x3_branch2 = Conv2D(filters=64, kernel_size=(3, 3), padding='same')
        self.conv_transpose_branch2 = Conv2DTranspose(filters=64, kernel_size=(2, 2), strides=(2, 2), padding='same')
        self.avg_pool_branch3 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')
        self.conv3x3_branch3 = Conv2D(filters=64, kernel_size=(3, 3), padding='same')
        self.conv_transpose_branch3 = Conv2DTranspose(filters=64, kernel_size=(2, 2), strides=(2, 2), padding='same')
        self.concat = Concatenate(axis=-1)
        self.conv1x1_fusion = Conv2D(filters=64, kernel_size=(1, 1), padding='same')

        # Branch path components
        self.conv1x1_branch = Conv2D(filters=64, kernel_size=(1, 1), padding='same')

        # Final classification layer
        self.dense = Dense(units=10, activation='softmax')

    def call(self, inputs):
        # Main path
        x = self.conv1x1_main(inputs)

        # First branch
        branch1 = self.conv3x3_branch1(x)

        # Second branch
        branch2 = self.avg_pool_branch2(x)
        branch2 = self.conv3x3_branch2(branch2)
        branch2 = self.conv_transpose_branch2(branch2)

        # Third branch
        branch3 = self.avg_pool_branch3(x)
        branch3 = self.conv3x3_branch3(branch3)
        branch3 = self.conv_transpose_branch3(branch3)

        # Concatenate branches
        concatenated = self.concat([branch1, branch2, branch3])
        main_output = self.conv1x1_fusion(concatenated)

        # Branch path
        branch_path = self.conv1x1_branch(inputs)

        # Fusion
        fused = tf.math.add(main_output, branch_path)

        # Final classification
        output = self.dense(tf.keras.layers.GlobalAveragePooling2D()(fused))

        return output

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal(shape=(1, 32, 32, 3))  # CIFAR-10 input shape
    output_tensor = model(input_tensor)
    print("Input shape:", input_tensor.shape)
    print("Output shape:", output_tensor.shape)