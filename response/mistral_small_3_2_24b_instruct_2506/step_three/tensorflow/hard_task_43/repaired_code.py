import tensorflow as tf
from tensorflow.keras.layers import Input, AveragePooling2D, Flatten, Concatenate, Dense, Reshape, Conv2D
from tensorflow.keras.models import Model

class DL_Model(Model):
    def __init__(self):
        super(DL_Model, self).__init__()
        # Block 1
        self.avg_pool_1x1 = AveragePooling2D(pool_size=(1, 1), strides=(1, 1), padding='valid')
        self.avg_pool_2x2 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')
        self.avg_pool_4x4 = AveragePooling2D(pool_size=(4, 4), strides=(4, 4), padding='valid')
        self.flatten_1 = Flatten()
        self.flatten_2 = Flatten()
        self.flatten_3 = Flatten()
        self.concatenate = Concatenate(axis=1)
        self.dense = Dense(units=128, activation='relu')
        self.reshape = Reshape(target_shape=(8, 8, 4))  # Changed to (8, 8, 4) to maintain dimensions

        # Block 2
        # Branch 1
        self.conv2d_1 = Conv2D(filters=4, kernel_size=(1, 1), strides=(1, 1), padding='valid', activation='relu')
        self.conv2d_2 = Conv2D(filters=4, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu')
        # Branch 2
        self.conv2d_3 = Conv2D(filters=4, kernel_size=(1, 1), strides=(1, 1), padding='valid', activation='relu')
        self.conv2d_4 = Conv2D(filters=4, kernel_size=(1, 7), strides=(1, 1), padding='valid', activation='relu')
        self.conv2d_5 = Conv2D(filters=4, kernel_size=(7, 1), strides=(1, 1), padding='valid', activation='relu')
        self.conv2d_6 = Conv2D(filters=4, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu')
        # Branch 3
        self.avg_pool_3 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')
        self.concatenate_2 = Concatenate(axis=-1)
        # Final layers
        self.dense_1 = Dense(units=128, activation='relu')
        self.dense_2 = Dense(units=10, activation='softmax')

    def call(self, inputs):
        # Block 1
        x1 = self.avg_pool_1x1(inputs)
        x2 = self.avg_pool_2x2(inputs)
        x3 = self.avg_pool_4x4(inputs)

        x1 = self.flatten_1(x1)
        x2 = self.flatten_2(x2)
        x3 = self.flatten_3(x3)

        x = self.concatenate([x1, x2, x3])
        x = self.dense(x)
        x = self.reshape(x)

        # Block 2
        # Branch 1
        branch1 = self.conv2d_1(x)
        branch1 = self.conv2d_2(branch1)
        # Branch 2
        branch2 = self.conv2d_3(x)
        branch2 = self.conv2d_4(branch2)
        branch2 = self.conv2d_5(branch2)
        branch2 = self.conv2d_6(branch2)
        # Branch 3
        branch3 = self.avg_pool_3(x)

        x = self.concatenate_2([branch1, branch2, branch3])
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = self.dense_1(x)
        x = self.dense_2(x)

        return x

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.keras.Input(shape=(28, 28, 1))
    output_tensor = model(input_tensor)
    model = Model(inputs=input_tensor, outputs=output_tensor)
    model.summary()