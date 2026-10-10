import tensorflow as tf

class DL_Model(tf.keras.Model):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Initial 1x1 convolution
        self.conv1 = tf.keras.layers.Conv2D(filters=32, kernel_size=1, strides=1, padding='same', activation='relu')

        # Branch 1: Local Feature Extraction
        self.conv2_1 = tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=1, padding='same', activation='relu')
        self.conv2_2 = tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=1, padding='same', activation='relu')

        # Branch 2: Downsampling and Upsampling
        self.avgpool2 = tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')
        self.conv3_1 = tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=1, padding='same', activation='relu')
        self.conv_transpose2 = tf.keras.layers.Conv2DTranspose(filters=32, kernel_size=3, strides=2, padding='same', activation='relu')

        # Branch 3: Downsampling and Upsampling
        self.avgpool3 = tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')
        self.conv4_1 = tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=1, padding='same', activation='relu')
        self.conv_transpose3 = tf.keras.layers.Conv2DTranspose(filters=32, kernel_size=3, strides=2, padding='same', activation='relu')

        # Concatenation and Refinement
        self.concatenate = tf.keras.layers.Concatenate(axis=-1)
        self.conv5 = tf.keras.layers.Conv2D(filters=64, kernel_size=1, strides=1, padding='same', activation='relu')

        # Classification
        self.flatten = tf.keras.layers.Flatten()
        self.dense = tf.keras.layers.Dense(units=10, activation='softmax')

    def call(self, inputs):
        x = self.conv1(inputs)

        # Branch 1
        branch1 = self.conv2_1(x)
        branch1 = self.conv2_2(branch1)

        # Branch 2
        branch2 = self.avgpool2(x)
        branch2 = self.conv3_1(branch2)
        branch2 = self.conv_transpose2(branch2)

        # Branch 3
        branch3 = self.avgpool3(x)
        branch3 = self.conv4_1(branch3)
        branch3 = self.conv_transpose3(branch3)

        # Concatenation
        merged = self.concatenate([branch1, branch2, branch3])
        merged = self.conv5(merged)

        # Classification
        flattened = self.flatten(merged)
        output = self.dense(flattened)

        return output


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal(shape=(1, 32, 32, 3))  # CIFAR-10 input shape
    output_tensor = model(input_tensor)
    print(output_tensor.shape)