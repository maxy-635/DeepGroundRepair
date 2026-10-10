import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Add, DepthwiseConv2D, Concatenate, Dense

class DL_Model(tf.keras.Model):
    def __init__(self):
        super(DL_Model, self).__init__()

        # First Block
        self.conv2d_main_1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', name='conv2d_main_1')
        self.conv2d_main_2 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', name='conv2d_main_2')
        self.conv2d_main_3 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', name='conv2d_main_3')
        self.add_1 = Add(name='add_1')

        # Second Block
        self.depthwiseconv2d_1 = DepthwiseConv2D(kernel_size=(1, 1), strides=(1, 1), padding='same', depth_multiplier=1, activation='relu', name='depthwiseconv2d_1')
        self.depthwiseconv2d_2 = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', depth_multiplier=1, activation='relu', name='depthwiseconv2d_2')
        self.depthwiseconv2d_3 = DepthwiseConv2D(kernel_size=(5, 5), strides=(1, 1), padding='same', depth_multiplier=1, activation='relu', name='depthwiseconv2d_3')
        self.concatenate_1 = Concatenate(axis=-1, name='concatenate_1')

        # Classifier
        self.dense_1 = Dense(units=128, activation='relu', name='dense_1')
        self.dense_2 = Dense(units=10, activation='softmax', name='dense_2')

    def call(self, inputs):
        # First Block
        main_path = self.conv2d_main_1(inputs)
        main_path = self.conv2d_main_2(main_path)
        main_path = self.conv2d_main_3(main_path)
        branch_path = inputs
        merged = self.add_1([main_path, branch_path])

        # Second Block
        group_1 = self.depthwiseconv2d_1(merged)
        group_2 = self.depthwiseconv2d_2(merged)
        group_3 = self.depthwiseconv2d_3(merged)
        concatenated = self.concatenate_1([group_1, group_2, group_3])

        # Classifier
        flattened = tf.keras.layers.Flatten()(concatenated)
        dense_1_output = self.dense_1(flattened)
        output = self.dense_2(dense_1_output)

        return output


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal((1, 32, 32, 3))  # Example input tensor (batch_size, height, width, channels)
    output_tensor = model(input_tensor)
    print("Output shape:", output_tensor.shape)