import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPool2D, Concatenate, Flatten, Dense, Add, Activation, BatchNormalization, ReLU, GlobalAveragePooling2D

class DL_Model(tf.keras.Model):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Branch 1: 3x3 convolutions
        self.branch1_conv1 = Conv2D(filters=64, kernel_size=3, strides=1, padding='same', activation='relu')
        self.branch1_conv2 = Conv2D(filters=64, kernel_size=3, strides=1, padding='same', activation='relu')

        # Branch 2: 1x1 convolutions followed by two 3x3 convolutions
        self.branch2_conv1 = Conv2D(filters=64, kernel_size=1, strides=1, padding='same', activation='relu')
        self.branch2_conv2 = Conv2D(filters=64, kernel_size=3, strides=1, padding='same', activation='relu')
        self.branch2_conv3 = Conv2D(filters=64, kernel_size=3, strides=1, padding='same', activation='relu')

        # Branch 3: Max pooling
        self.branch3_pool = MaxPool2D(pool_size=2, strides=2, padding='same')
        self.branch3_conv1 = Conv2D(filters=64, kernel_size=3, strides=1, padding='same', activation='relu')

        # Multi-scale feature fusion
        self.concatenate = Concatenate(axis=-1)

        # Classification head
        self.global_avg_pool = GlobalAveragePooling2D()
        self.dense1 = Dense(units=128, activation='relu')
        self.dense2 = Dense(units=10, activation='softmax')

    def call(self, inputs):
        # Branch 1 processing
        x1 = self.branch1_conv1(inputs)
        x1 = self.branch1_conv2(x1)

        # Branch 2 processing
        x2 = self.branch2_conv1(inputs)
        x2 = self.branch2_conv2(x2)
        x2 = self.branch2_conv3(x2)

        # Branch 3 processing
        x3 = self.branch3_pool(inputs)
        x3 = self.branch3_conv1(x3)

        # Multi-scale feature fusion
        fused_features = self.concatenate([x1, x2, x3])

        # Classification head
        x = self.global_avg_pool(fused_features)
        x = self.dense1(x)
        x = self.dense2(x)

        return x

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal(shape=(1, 32, 32, 3))  # CIFAR-10 image shape: 32x32x3
    output_tensor = model(input_tensor)
    print("Output shape:", output_tensor.shape)