import tensorflow as tf
from tensorflow.keras.layers import Conv2D, DepthwiseConv2D, Conv2DTranspose, Dense, Reshape, Concatenate, Lambda, Add, Permute, GlobalAveragePooling2D
from tensorflow.keras.models import Model

class DL_Model(Model):
    def __init__(self):
        super(DL_Model, self).__init__()
        # Block 1 components
        self.conv1_1 = Conv2D(filters=64, kernel_size=3, strides=1, padding='same', activation='relu')
        self.conv1_2 = Conv2D(filters=64, kernel_size=3, strides=1, padding='same', activation='relu')
        self.conv1_3 = Conv2D(filters=64, kernel_size=3, strides=1, padding='same', activation='relu')

        # Block 2 components
        self.permute = Permute(dims=(1, 2, 4, 3))
        self.reshape = Reshape(target_shape=(None, None, 3, -1))

        # Block 3 components
        self.depthwise_conv = DepthwiseConv2D(kernel_size=3, strides=1, padding='same', depth_multiplier=1, activation='relu')
        self.pointwise_conv = Conv2D(filters=192, kernel_size=1, strides=1, padding='same', activation='relu')

        # Branch components
        self.branch_conv = Conv2D(filters=192, kernel_size=1, strides=1, padding='same', activation='relu')

        # Final components
        self.add = Add()
        self.global_avg_pool = GlobalAveragePooling2D()
        self.dense = Dense(units=10, activation='softmax')

    def call(self, inputs):
        # Main path
        x = inputs

        # Block 1: Channel splitting and fusion
        x1 = self.conv1_1(x)
        x2 = self.conv1_2(x)
        x3 = self.conv1_3(x)

        # Split channels into three groups
        channels = x.shape[-1]
        x1 = Conv2D(filters=channels//3, kernel_size=1, strides=1, padding='same')(x1)
        x2 = Conv2D(filters=channels//3, kernel_size=1, padding='same')(x2)
        x3 = Conv2D(filters=channels//3, kernel_size=1, padding='same')(x3)

        # Concatenate along channel dimension
        x = Concatenate(axis=-1)([x1, x2, x3])

        # Block 2: Channel shuffling
        h, w, c = x.shape[1], x.shape[2], x.shape[3]
        x = Reshape(target_shape=(h, w, 3, c//3))(x)
        x = self.permute(x)
        x = Reshape(target_shape=(h, w, c))(x)

        # Block 3: Depthwise separable convolution
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)

        # Branch path
        branch = self.branch_conv(inputs)

        # Combine main path and branch
        x = self.add([x, branch])

        # Final classification
        x = self.global_avg_pool(x)
        x = self.dense(x)

        return x

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.keras.Input(shape=(32, 32, 3))  # CIFAR-10 image shape
    output_tensor = model(input_tensor)
    print("Model created successfully!")
    print("Input shape:", input_tensor.shape)
    print("Output shape:", output_tensor.shape)