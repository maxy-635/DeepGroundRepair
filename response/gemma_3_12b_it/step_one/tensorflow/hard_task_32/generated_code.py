import tensorflow as tf
from tensorflow.keras.layers import DepthwiseConv2D, Conv2D, Dropout, concatenate, Dense, Input

class DL_Model(tf.keras.Model):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Define the specialized block
        def specialized_block(input_tensor):
            # Depthwise separable convolution
            depthwise_conv = DepthwiseConv2D(
                kernel_size=(3, 3),
                strides=(1, 1),
                padding='same',
                depth_multiplier=1,
                activation='relu',
                name='depthwise_conv'
            )(input_tensor)

            # 1x1 convolution
            pointwise_conv = Conv2D(
                filters=64,
                kernel_size=(1, 1),
                strides=(1, 1),
                padding='same',
                activation='relu',
                name='pointwise_conv'
            )(depthwise_conv)

            # Dropout
            dropout1 = Dropout(
                rate=0.2,
                name='dropout1'
            )(pointwise_conv)

            # Dropout after the second convolution
            dropout2 = Dropout(
                rate=0.2,
                name='dropout2'
            )(depthwise_conv)

            return concatenate([dropout1, dropout2], axis=-1)

        # Three branches
        input_tensor = Input(shape=(28, 28, 1), name='input_tensor')
        branch1 = specialized_block(input_tensor)
        branch2 = specialized_block(tf.keras.layers.Conv2D(
            filters=32,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='same',
            activation='relu',
            name='branch2_conv1'
        )(input_tensor))
        branch3 = specialized_block(tf.keras.layers.Conv2D(
            filters=32,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='same',
            activation='relu',
            name='branch3_conv1'
        )(input_tensor))

        # Concatenate the outputs of the three branches
        merged = concatenate([branch1, branch2, branch3], axis=-1)

        # Fully connected layers
        dense1 = Dense(
            units=128,
            activation='relu',
            name='dense1'
        )(merged)

        output_tensor = Dense(
            units=10,
            activation='softmax',
            name='output_tensor'
        )(dense1)

        self.build(input_shape=(None, 28, 28, 1))
        self.input = input_tensor
        self.output = output_tensor


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal((1, 28, 28, 1))
    output_tensor = model(input_tensor)
    print(output_tensor.shape)