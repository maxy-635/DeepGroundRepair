import tensorflow as tf

class DL_Model(tf.keras.Model):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Block 1
        self.conv1_1 = tf.keras.layers.SeparableConv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')
        self.bn1_1 = tf.keras.layers.BatchNormalization(axis=-1)
        self.conv1_2 = tf.keras.layers.SeparableConv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')
        self.bn1_2 = tf.keras.layers.BatchNormalization(axis=-1)
        self.conv1_3 = tf.keras.layers.SeparableConv2D(filters=32, kernel_size=(5, 5), padding='same', activation='relu')
        self.bn1_3 = tf.keras.layers.BatchNormalization(axis=-1)
        self.concat1 = tf.keras.layers.Concatenate(axis=-1)

        # Block 2
        self.conv2_1 = tf.keras.layers.Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')
        self.avgpool2 = tf.keras.layers.AveragePooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')
        self.conv2_2 = tf.keras.layers.Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')
        self.conv2_3 = tf.keras.layers.Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')
        self.conv2_4_1 = tf.keras.layers.Conv2D(filters=32, kernel_size=(1, 3), padding='same', activation='relu')
        self.conv2_4_2 = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 1), padding='same', activation='relu')
        self.concat2_1 = tf.keras.layers.Concatenate(axis=-1)

        self.conv2_5 = tf.keras.layers.Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')
        self.conv2_6 = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')
        self.conv2_7_1 = tf.keras.layers.Conv2D(filters=32, kernel_size=(1, 3), padding='same', activation='relu')
        self.conv2_7_2 = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 1), padding='same', activation='relu')
        self.concat2_2 = tf.keras.layers.Concatenate(axis=-1)

        self.concat2 = tf.keras.layers.Concatenate(axis=-1)

        # Classification
        self.flatten = tf.keras.layers.Flatten()
        self.dense = tf.keras.layers.Dense(units=10, activation='softmax')

    def call(self, inputs):
        # Block 1
        x1_1 = self.conv1_1(inputs)
        x1_1 = self.bn1_1(x1_1)
        x1_2 = self.conv1_2(inputs)
        x1_2 = self.bn1_2(x1_2)
        x1_3 = self.conv1_3(inputs)
        x1_3 = self.bn1_3(x1_3)
        x1 = self.concat1([x1_1, x1_2, x1_3])

        # Block 2
        x2_1 = self.conv2_1(x1)
        x2_2 = self.conv2_2(self.avgpool2(x1))
        x2_3 = self.conv2_3(x1)
        x2_4_1 = self.conv2_4_1(x2_3)
        x2_4_2 = self.conv2_4_2(x2_3)
        x2_4 = self.concat2_1([x2_4_1, x2_4_2])

        x2_5 = self.conv2_5(x1)
        x2_6 = self.conv2_6(x2_5)
        x2_7_1 = self.conv2_7_1(x2_6)
        x2_7_2 = self.conv2_7_2(x2_6)
        x2_7 = self.concat2_2([x2_7_1, x2_7_2])

        x2 = self.concat2([x2_1, x2_2, x2_4, x2_7])

        # Classification
        x = self.flatten(x2)
        return self.dense(x)


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal(shape=(1, 32, 32, 3))
    output_tensor = model(input_tensor)
    print(output_tensor.shape)