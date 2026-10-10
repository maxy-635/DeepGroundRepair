import tensorflow as tf
from tensorflow.keras.layers import Conv2D, AveragePooling2D, Conv2DTranspose, Concatenate, Dense, Add

class DL_Model(tf.keras.Model):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Main Path
        self.conv1x1_main = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu', name="conv1x1_main")
        self.branch1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', name="branch1")

        self.avg_pool1 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same', name="avg_pool1")
        self.conv1x1_branch2 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu', name="conv1x1_branch2")
        self.conv_branch2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', name="conv_branch2")
        self.conv1x1_transpose_branch2 = Conv2DTranspose(filters=64, kernel_size=(2, 2), strides=(2, 2), padding='same', activation='relu', name="conv1x1_transpose_branch2")

        self.avg_pool2 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same', name="avg_pool2")
        self.conv1x1_branch3 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu', name="conv1x1_branch3")
        self.conv_branch3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', name="conv_branch3")
        self.conv1x1_transpose_branch3 = Conv2DTranspose(filters=64, kernel_size=(2, 2), strides=(2, 2), padding='same', activation='relu', name="conv1x1_transpose_branch3")

        self.concat = Concatenate(axis=-1, name="concat")
        self.conv1x1_output = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu', name="conv1x1_output")

        # Branch Path
        self.conv1x1_branch = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu', name="conv1x1_branch")

        self.add = Add(name="add")
        self.dense = Dense(units=10, activation='softmax', name="dense")

    def call(self, inputs):
        # Main Path
        main_path = self.conv1x1_main(inputs)
        branch1_output = self.branch1(main_path)

        main_path_downsampled1 = self.avg_pool1(main_path)
        branch2_output = self.conv1x1_branch2(main_path_downsampled1)
        branch2_output = self.conv_branch2(branch2_output)
        branch2_output = self.conv1x1_transpose_branch2(branch2_output)

        main_path_downsampled2 = self.avg_pool2(main_path)
        branch3_output = self.conv1x1_branch3(main_path_downsampled2)
        branch3_output = self.conv_branch3(branch3_output)
        branch3_output = self.conv1x1_transpose_branch3(branch3_output)

        merged_output = self.concat([branch1_output, branch2_output, branch3_output])
        main_path_output = self.conv1x1_output(merged_output)

        # Branch Path
        branch_path = self.conv1x1_branch(inputs)

        # Fusion
        fused_output = self.add([main_path_output, branch_path])

        # Classification
        output = self.dense(tf.keras.layers.Flatten()(fused_output))

        return output


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal((1, 32, 32, 3))  # Example input shape for CIFAR-10
    output_tensor = model(input_tensor)
    print(output_tensor.shape)