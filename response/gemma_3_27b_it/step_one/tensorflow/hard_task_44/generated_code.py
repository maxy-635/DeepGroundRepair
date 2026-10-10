import tensorflow as tf

class DL_Model(tf.keras.Model):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Block 1
        self.conv1_1 = tf.keras.layers.Conv2D(filters=32, kernel_size=1, strides=1, padding='same', activation='relu')
        self.conv1_3 = tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=1, padding='same', activation='relu')
        self.conv1_5 = tf.keras.layers.Conv2D(filters=32, kernel_size=5, strides=1, padding='same', activation='relu')
        self.dropout1 = tf.keras.layers.Dropout(rate=0.25)
        self.concat1 = tf.keras.layers.Concatenate(axis=-1)

        # Block 2
        self.branch1 = tf.keras.layers.Conv2D(filters=32, kernel_size=1, strides=1, padding='same', activation='relu')
        self.branch2_conv1 = tf.keras.layers.Conv2D(filters=32, kernel_size=1, strides=1, padding='same', activation='relu')
        self.branch2_conv3 = tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=1, padding='same', activation='relu')
        self.branch3_conv1 = tf.keras.layers.Conv2D(filters=32, kernel_size=1, strides=1, padding='same', activation='relu')
        self.branch3_conv5 = tf.keras.layers.Conv2D(filters=32, kernel_size=5, strides=1, padding='same', activation='relu')
        self.branch4_pool = tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=(1, 1), padding='same')
        self.branch4_conv1 = tf.keras.layers.Conv2D(filters=32, kernel_size=1, strides=1, padding='same', activation='relu')
        self.concat2 = tf.keras.layers.Concatenate(axis=-1)

        # Final layers
        self.flatten = tf.keras.layers.Flatten()
        self.dense = tf.keras.layers.Dense(units=10, activation='softmax')

    def call(self, inputs):
        # Block 1
        x1_1 = self.conv1_1(inputs)
        x1_3 = self.conv1_3(inputs)
        x1_5 = self.conv1_5(inputs)
        x1_drop = self.dropout1(x1_1)
        x1 = self.concat1([x1_drop, x1_3, x1_5])

        # Block 2
        x2_1 = self.branch1(x1)
        x2_2 = self.concat2([self.branch2_conv1(x1), self.branch2_conv3(x1)])
        x2_3 = self.concat2([self.branch3_conv1(x1), self.branch3_conv5(x1)])
        x2_4 = self.concat2([self.branch4_pool(x1), self.branch4_conv1(x1)])
        x2 = self.concat2([x2_1, x2_2, x2_3, x2_4])

        # Final layers
        x = self.flatten(x2)
        output = self.dense(x)

        return output


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal((1, 32, 32, 3))
    output_tensor = model(input_tensor)
    print(output_tensor.shape)