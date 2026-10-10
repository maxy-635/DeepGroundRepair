import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Dropout, Add, DepthwiseConv2D, Concatenate, Flatten, Dense

class DL_Model(tf.keras.Model):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Block 1
        self.conv1_main = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')
        self.dropout1 = Dropout(rate=0.2)
        self.conv1_restore = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation=None)  # No activation here

        self.conv2_branch = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')

        # Block 2
        self.depthwise_conv1 = DepthwiseConv2D(kernel_size=(1, 1), strides=(1, 1), padding='same', depth_multiplier=1, activation='relu')
        self.dropout2_1 = Dropout(rate=0.2)
        self.depthwise_conv2 = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', depth_multiplier=1, activation='relu')
        self.dropout2_2 = Dropout(rate=0.2)
        self.depthwise_conv3 = DepthwiseConv2D(kernel_size=(5, 5), strides=(1, 1), padding='same', depth_multiplier=1, activation='relu')
        self.dropout2_3 = Dropout(rate=0.2)


        # Final layers
        self.flatten = Flatten()
        self.dense = Dense(units=10, activation='softmax')

    def call(self, inputs):
        # Block 1
        main_path = self.conv1_main(inputs)
        main_path = self.dropout1(main_path)
        main_path = self.conv1_restore(main_path)

        branch_path = self.conv2_branch(inputs)

        merged_output = Add()([main_path, branch_path])

        # Block 2
        group1 = self.depthwise_conv1(merged_output)
        group1 = self.dropout2_1(group1)
        group2 = self.depthwise_conv2(merged_output)
        group2 = self.dropout2_2(group2)
        group3 = self.depthwise_conv3(merged_output)
        group3 = self.dropout2_3(group3)

        concatenated_output = Concatenate(axis=-1)([group1, group2, group3])

        # Final layers
        flattened_output = self.flatten(concatenated_output)
        output = self.dense(flattened_output)

        return output


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal((1, 32, 32, 3))  # CIFAR-10 input shape: (batch_size, height, width, channels)
    output_tensor = model(input_tensor)
    print("Output shape:", output_tensor.shape)