import tensorflow as tf
from tensorflow.keras.layers import Lambda, Conv2D, DepthwiseConv2D, SeparableConv2D, Concatenate, Add, Dense, Flatten
from tensorflow.keras.models import Model

class DL_Model(Model):
    def __init__(self):
        super(DL_Model, self).__init__()
        # Main path components
        self.conv1x1 = SeparableConv2D(filters=64, kernel_size=(1, 1), padding='same', activation='relu')
        self.conv3x3 = SeparableConv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')
        self.conv5x5 = SeparableConv2D(filters=64, kernel_size=(5, 5), padding='same', activation='relu')
        self.concatenate = Concatenate(axis=-1)

        # Branch path component
        self.branch_conv = Conv2D(filters=192, kernel_size=(1, 1), padding='same', activation='relu')

        # Final layers
        self.add = Add()
        self.flatten = Flatten()
        self.dense1 = Dense(units=512, activation='relu')
        self.dense2 = Dense(units=10, activation='softmax')

    def call(self, inputs):
        # Split input into three groups along the channel axis
        input_shape = tf.shape(inputs)
        split_size = input_shape[-1] // 3
        x1 = Lambda(lambda x: x[:, :, :, :split_size])(inputs)
        x2 = Lambda(lambda x: x[:, :, :, split_size:2*split_size])(inputs)
        x3 = Lambda(lambda x: x[:, :, :, 2*split_size:])(inputs)

        # Main path
        x1 = self.conv1x1(x1)
        x2 = self.conv3x3(x2)
        x3 = self.conv5x5(x3)
        main_output = self.concatenate([x1, x2, x3])

        # Branch path
        branch_output = self.branch_conv(inputs)

        # Combine main and branch paths
        combined = self.add([main_output, branch_output])

        # Final classification layers
        x = self.flatten(combined)
        x = self.dense1(x)
        output = self.dense2(x)

        return output

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal(shape=(1, 32, 32, 3))  # CIFAR-10 input shape
    output_tensor = model(input_tensor)
    print("Model created successfully. Output shape:", output_tensor.shape)