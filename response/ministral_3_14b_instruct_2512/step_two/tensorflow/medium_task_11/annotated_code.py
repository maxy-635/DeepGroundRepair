import tensorflow as tf
from tensorflow.keras.layers import Conv2D, GlobalAveragePooling2D, GlobalMaxPool2D, Dense, Add, Activation, Multiply, Concatenate, Flatten, Lambda
from tensorflow.keras.models import Model

class DL_Model(Model):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Initial convolutional layer
        self.conv_initial = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')

        # Channel attention path
        self.avg_pool = GlobalAveragePooling2D(keepdims=True)
        self.max_pool = GlobalMaxPool2D(keepdims=True)
        self.fc1 = Dense(units=128, activation='relu')
        self.fc2 = Dense(units=64, activation='relu')
        self.add = Add()
        self.activation = Activation('sigmoid')

        # Spatial attention path
        self.avg_pool_spatial = GlobalAveragePooling2D()
        self.max_pool_spatial = GlobalMaxPool2D()
        self.concat = Concatenate(axis=-1)

        # Final layers
        self.multiply = Multiply()
        self.flatten = Flatten()
        self.fc_final = Dense(units=10, activation='softmax')

    def call(self, inputs):
        # the initial shape of inputs is: (1, 32, 32, 3)
        # Initial convolution
        x = self.conv_initial(inputs)  # output tensor shape: (1, 32, 32, 64)

        # Channel attention
        avg = self.avg_pool(x)  # output tensor shape: (1, 1, 1, 64)
        max = self.max_pool(x)  # output tensor shape: (1, 1, 1, 64)
        avg_fc = self.fc1(avg)  # output tensor shape: (1, 1, 1, 128)
        max_fc = self.fc1(max)  # output tensor shape: (1, 1, 1, 128)
        avg_fc = self.fc2(avg_fc)  # output tensor shape: (1, 1, 1, 64)
        max_fc = self.fc2(max_fc)  # output tensor shape: (1, 1, 1, 64)
        channel_attention = self.add([avg_fc, max_fc])  # output tensor shape: (1, 1, 1, 64)
        channel_attention = self.activation(channel_attention)  # output tensor shape: (1, 1, 1, 64)
        channel_attention = tf.expand_dims(tf.expand_dims(channel_attention, axis=1), axis=1)  # output tensor shape: (1, 1, 1, 1, 1, 64)
        x = Multiply()([x, channel_attention])  # output tensor shape: (1, 1, 1, 32, 32, 64)

        # Spatial attention
        avg_spatial = self.avg_pool_spatial(x)
        max_spatial = self.max_pool_spatial(x)
        spatial_attention = self.concat([avg_spatial, max_spatial])
        spatial_attention = tf.expand_dims(spatial_attention, axis=-1)
        spatial_attention = tf.expand_dims(spatial_attention, axis=-1)
        x = Multiply()([x, spatial_attention])

        # Final layers
        x = self.flatten(x)
        output = self.fc_final(x)

        return output

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal(shape=(1, 32, 32, 3))  # CIFAR-10 image shape
    output_tensor = model(input_tensor)
    print("Model output shape:", output_tensor.shape)