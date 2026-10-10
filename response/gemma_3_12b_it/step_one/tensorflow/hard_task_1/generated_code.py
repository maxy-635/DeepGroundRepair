import tensorflow as tf
from tensorflow.keras.layers import Conv2D, GlobalAveragePooling2D, GlobalMaxPool2D, Dense, Add, Activation, Concatenate, Conv1D
from tensorflow.keras.models import Model
from tensorflow.keras.backend import sigmoid
from tensorflow.keras.backend import multiply

class DL_Model(Model):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Initial convolutional layer
        self.conv_initial = Conv2D(
            filters=32,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='same',
            activation='relu'
        )

        # Block 1: Channel Attention
        self.gap = GlobalAveragePooling2D(keepdims=False)
        self.gmp = GlobalMaxPool2D(keepdims=False)
        self.dense1_path1 = Dense(
            units=32,
            activation='relu'
        )
        self.dense2_path1 = Dense(
            units=32,
            activation='relu'
        )
        self.dense1_path2 = Dense(
            units=32,
            activation='relu'
        )
        self.dense2_path2 = Dense(
            units=32,
            activation='relu'
        )
        self.add_channel_attention = Add()
        self.activation_channel_attention = Activation('relu')
        self.multiply_channel_attention = multiply

        # Block 2: Spatial Attention
        self.avg_pool = GlobalAveragePooling2D(keepdims=True)
        self.max_pool = GlobalMaxPool2D(keepdims=True)
        self.concat = Concatenate(axis=-1)
        self.conv1d_spatial = Conv1D(
            filters=1,
            kernel_size=(1, 1),
            strides=(1, 1),
            padding='same',
            activation=sigmoid
        )
        self.multiply_spatial_attention = multiply

        # Additional branch
        self.conv_additional = Conv2D(
            filters=32,
            kernel_size=(1, 1),
            strides=(1, 1),
            padding='same',
            activation='relu'
        )
        self.add_additional = Add()

        # Final classification
        self.dense_final = Dense(
            units=10,
            activation='softmax'
        )

    def call(self, inputs):
        x = self.conv_initial(inputs)

        # Block 1: Channel Attention
        path1 = self.gap(x)
        path1 = self.dense1_path1(path1)
        path1 = self.dense2_path1(path1)

        path2 = self.gmp(x)
        path2 = self.dense1_path2(path2)
        path2 = self.dense2_path2(path2)

        channel_attention = self.add_channel_attention([path1, path2])
        channel_attention = self.activation_channel_attention(channel_attention)
        x = self.multiply_channel_attention([x, channel_attention])

        # Block 2: Spatial Attention
        avg = self.avg_pool(x)
        max_ = self.max_pool(x)
        spatial_attention = self.concat([avg, max_])
        spatial_attention = self.conv1d_spatial(spatial_attention)
        x = self.multiply_spatial_attention([x, spatial_attention])

        # Additional branch
        additional_branch = self.conv_additional(x)
        x = self.add_additional([x, additional_branch])

        # Final classification
        x = self.dense_final(x)

        return x


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal((1, 32, 32, 3))  # Dummy input tensor (batch_size, height, width, channels)
    output_tensor = model(input_tensor)
    print("Output shape:", output_tensor.shape)