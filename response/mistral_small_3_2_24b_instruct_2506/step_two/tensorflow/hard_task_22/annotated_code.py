import tensorflow as tf
from tensorflow.keras.layers import Lambda, Conv2D, DepthwiseConv2D, SeparableConv2D, Concatenate, Add, Flatten, Dense
from tensorflow.keras.models import Model

class DL_Model(Model):
    def __init__(self):
        super(DL_Model, self).__init__()
        # Main path components
        self.conv1x1 = SeparableConv2D(filters=64, kernel_size=(1, 1), padding='same', activation='relu')
        self.conv3x3 = SeparableConv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')
        self.conv5x5 = SeparableConv2D(filters=64, kernel_size=(5, 5), padding='same', activation='relu')
        self.concat = Concatenate(axis=-1)

        # Branch path component
        self.branch_conv = Conv2D(filters=192, kernel_size=(1, 1), padding='same', activation='relu')

        # Fusion and classification components
        self.add = Add()
        self.flatten = Flatten()
        self.dense1 = Dense(units=512, activation='relu')
        self.dense2 = Dense(units=10, activation='softmax')

    def call(self, inputs):
        # Split input into three groups along channel axis
        input_shape = tf.shape(inputs)
        split_size = input_shape[-1] // 3
        x1 = Lambda(lambda x: x[..., :split_size])(inputs)
        x2 = Lambda(lambda x: x[..., split_size:2*split_size])(inputs)
        x3 = Lambda(lambda x: x[..., 2*split_size:])(inputs)

        # Main path processing
        x1 = self.conv1x1(x1)
        x2 = self.conv3x3(x2)
        x3 = self.conv5x5(x3)
        main_output = self.concat([x1, x2, x3])

        # Branch path processing
        branch_output = self.branch_conv(inputs)

        # Fusion
        fused_output = self.add([main_output, branch_output])

        # Classification
        flattened = self.flatten(fused_output)
        output = self.dense1(flattened)
        output = self.dense2(output)

        return output

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.keras.Input(shape=(32, 32, 3))  # CIFAR-10 image shape
    output_tensor = model(input_tensor)
    print(output_tensor)