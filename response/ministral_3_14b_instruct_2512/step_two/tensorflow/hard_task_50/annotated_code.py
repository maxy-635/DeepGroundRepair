import tensorflow as tf
from tensorflow.keras.layers import MaxPool2D, Flatten, Dropout, Concatenate, Dense, Reshape, SeparableConv2D, GlobalAveragePooling2D
from tensorflow.keras.models import Model

class DL_Model(Model):
    def __init__(self):
        super(DL_Model, self).__init__()

        # First block
        self.max_pool_1x1 = MaxPool2D(pool_size=(1, 1), strides=(1, 1), padding='valid')
        self.max_pool_2x2 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid')
        self.max_pool_4x4 = MaxPool2D(pool_size=(4, 4), strides=(4, 4), padding='valid')

        self.flatten_1x1 = Flatten()
        self.flatten_2x2 = Flatten()
        self.flatten_4x4 = Flatten()

        self.dropout = Dropout(rate=0.5)

        self.dense = Dense(units=1024, activation='relu')
        self.reshape = Reshape(target_shape=(1, 1, 1024))

        # Second block
        self.separable_conv_1x1 = SeparableConv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')
        self.separable_conv_3x3 = SeparableConv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')
        self.separable_conv_5x5 = SeparableConv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')
        self.separable_conv_7x7 = SeparableConv2D(filters=64, kernel_size=(7, 7), strides=(1, 1), padding='same', activation='relu')

        self.concatenate = Concatenate(axis=-1)

        # Final classification
        self.global_avg_pool = GlobalAveragePooling2D()
        self.final_dense = Dense(units=10, activation='softmax')

    def call(self, inputs):
        # First block processing
        pool_1x1 = self.max_pool_1x1(inputs)
        pool_2x2 = self.max_pool_2x2(inputs)
        pool_4x4 = self.max_pool_4x4(inputs)

        flat_1x1 = self.flatten_1x1(pool_1x1)
        flat_2x2 = self.flatten_2x2(pool_2x2)
        flat_4x4 = self.flatten_4x4(pool_4x4)

        # Concatenate with dropout
        concatenated = Concatenate(axis=-1)([flat_1x1, flat_2x2, flat_4x4])
        concatenated = self.dropout(concatenated)

        # Transform for second block
        transformed = self.dense(concatenated)
        transformed = self.reshape(transformed)

        # Second block processing
        # Split into four groups along last dimension
        split = tf.split(transformed, num_or_size_splits=4, axis=-1)

        # Process each group with different kernel sizes
        conv_1x1 = self.separable_conv_1x1(split[0])
        conv_3x3 = self.separable_conv_3x3(split[1])
        conv_5x5 = self.separable_conv_5x5(split[2])
        conv_7x7 = self.separable_conv_7x7(split[3])

        # Concatenate outputs
        concatenated_conv = self.concatenate([conv_1x1, conv_3x3, conv_5x5, conv_7x7])

        # Final classification
        pooled = self.global_avg_pool(concatenated_conv)
        output = self.final_dense(pooled)

        return output

if __name__ == "__main__":
    model = DL_Model()

    # Create a dummy input tensor with shape (batch_size, height, width, channels)
    # CIFAR-10 images are 32x32 with 3 channels
    input_tensor = tf.random.normal(shape=(1, 32, 32, 3))

    output_tensor = model(input_tensor)
    print("Output shape:", output_tensor.shape)