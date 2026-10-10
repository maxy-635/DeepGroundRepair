import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, ReLU, Add, GlobalAveragePooling2D, Flatten, Dense

class DL_Model(tf.keras.Model):
    def __init__(self):
        super(DL_Model, self).__init__()
        # Initial convolutional layer to reduce dimensionality
        self.conv1 = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same')

        # First basic block
        self.basic_block1_conv = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same')
        self.basic_block1_bn = BatchNormalization(axis=-1)
        self.basic_block1_relu = ReLU()

        # Second basic block
        self.basic_block2_conv = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same')
        self.basic_block2_bn = BatchNormalization(axis=-1)
        self.basic_block2_relu = ReLU()

        # Branch convolutional layer
        self.branch_conv = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='same')

        # Global average pooling and dense layer
        self.global_avg_pool = GlobalAveragePooling2D(keepdims=False)
        self.dense = Dense(units=10, activation='softmax')

    def call(self, inputs):
        # Initial convolutional layer
        x = self.conv1(inputs)

        # First basic block
        main_path = self.basic_block1_conv(x)
        main_path = self.basic_block1_bn(main_path)
        main_path = self.basic_block1_relu(main_path)

        # Branch path
        branch_path = self.branch_conv(x)

        # Feature fusion by addition
        x = Add()([main_path, branch_path])

        # Second basic block
        main_path = self.basic_block2_conv(x)
        main_path = self.basic_block1_bn(main_path)
        main_path = self.basic_block1_relu(main_path)

        # Branch path
        branch_path = self.branch_conv(x)

        # Feature fusion by addition
        x = Add()([main_path, branch_path])

        # Global average pooling and dense layer
        x = self.global_avg_pool(x)
        x = self.dense(x)

        return x

if __name__ == "__main__":
    # Create a dummy input tensor with shape (batch_size, height, width, channels)
    input_tensor = tf.random.normal(shape=(1, 32, 32, 3))
    output_tensor = DL_Model()(input_tensor)
    print("Input shape:", input_tensor.shape)
    print("Output shape:", output_tensor.shape)