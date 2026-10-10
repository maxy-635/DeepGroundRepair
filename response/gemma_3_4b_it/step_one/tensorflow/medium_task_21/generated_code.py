import tensorflow as tf

class DL_Model(tf.keras.Model):
    def __init__(self):
        super(DL_Model, self).__init__()
        self.branch1 = tf.keras.layers.Conv2D(filters=32, kernel_size=1, strides=1, padding='same')
        self.bn1 = tf.keras.layers.BatchNormalization(axis=-1)
        self.dropout1 = tf.keras.layers.Dropout(rate=0.25)

        self.branch2 = tf.keras.layers.Conv2D(filters=32, kernel_size=1, strides=1, padding='same')
        self.bn2 = tf.keras.layers.BatchNormalization(axis=-1)
        self.dropout2 = tf.keras.layers.Dropout(rate=0.25)

        self.branch3 = tf.keras.layers.Conv2D(filters=32, kernel_size=1, strides=1, padding='same')
        self.bn3 = tf.keras.layers.BatchNormalization(axis=-1)
        self.dropout3 = tf.keras.layers.Dropout(rate=0.25)

        self.branch4 = tf.keras.layers.Conv2D(filters=32, kernel_size=1, strides=1, padding='same')
        self.bn4 = tf.keras.layers.BatchNormalization(axis=-1)
        self.dropout4 = tf.keras.layers.Dropout(rate=0.25)

        self.avg_pool = tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding='valid')
        self.conv_branch4 = tf.keras.layers.Conv2D(filters=16, kernel_size=1, strides=1, padding='same')
        self.bn_branch4 = tf.keras.layers.BatchNormalization(axis=-1)

        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(units=64, activation='relu')
        self.dropout_dense1 = tf.keras.layers.Dropout(rate=0.5)
        self.dense2 = tf.keras.layers.Dense(units=64, activation='relu')
        self.dropout_dense2 = tf.keras.layers.Dropout(rate=0.5)
        self.dense3 = tf.keras.layers.Dense(units=10, activation='softmax')

    def call(self, inputs):
        branch1 = self.branch1(inputs)
        branch1 = self.bn1(branch1)
        branch1 = self.dropout1(branch1)

        branch2 = self.branch2(inputs)
        branch2 = self.bn2(branch2)
        branch2 = self.dropout2(branch2)

        branch3 = self.branch3(inputs)
        branch3 = self.bn3(branch3)
        branch3 = self.dropout3(branch3)

        branch4 = self.branch4(inputs)
        branch4 = self.bn4(branch4)
        branch4 = self.dropout4(branch4)
        branch4 = self.conv_branch4(branch4)
        branch4 = self.bn_branch4(branch4)

        concatenated = tf.keras.layers.Concatenate(axis=-1)([branch1, branch2, branch3, branch4])
        flattened = self.flatten(concatenated)
        dense1 = self.dense1(flattened)
        dense1 = self.dropout_dense1(dense1)
        dense2 = self.dense2(dense1)
        dense2 = self.dropout_dense2(dense2)
        outputs = self.dense3(dense2)

        return outputs

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal((1, 32, 32, 3))
    output_tensor = model(input_tensor)
    print(output_tensor.shape)