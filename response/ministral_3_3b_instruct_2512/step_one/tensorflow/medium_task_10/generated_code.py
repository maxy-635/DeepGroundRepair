import tensorflow as tf

class DL_Model(tf.keras.Model):
    def __init__(self):
        super(DL_Model, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same')
        self.bn1 = tf.keras.layers.BatchNormalization(axis=-1)
        self.relu1 = tf.keras.layers.ReLU(max_value=10.0)
        self.add1 = tf.keras.layers.Add()

        self.conv2 = tf.keras.layers.Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same')
        self.bn2 = tf.keras.layers.BatchNormalization(axis=-1)
        self.relu2 = tf.keras.layers.ReLU(max_value=10.0)
        self.conv_branch2 = tf.keras.layers.Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='same')
        self.add2 = tf.keras.layers.Add()

        self.conv3 = tf.keras.layers.Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same')
        self.bn3 = tf.keras.layers.BatchNormalization(axis=-1)
        self.relu3 = tf.keras.layers.ReLU(max_value=10.0)
        self.conv_branch3 = tf.keras.layers.Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='same')
        self.add3 = tf.keras.layers.Add()

        self.global_avg_pool = tf.keras.layers.GlobalAveragePooling2D(keepdims=False)
        self.dense = tf.keras.layers.Dense(units=10, activation='softmax')

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.relu1(x)
        x_branch = x
        x = self.add1([x, x_branch])
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x_branch = self.conv_branch2(x)
        x = self.add2([x, x_branch])
        x = self.relu2(x)

        x_branch = self.conv1(inputs)
        x_branch = self.bn1(x_branch)
        x_branch = self.relu1(x_branch)
        x_branch = self.add1([x_branch, x_branch])
        x_branch = self.relu1(x_branch)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x_branch = self.conv_branch3(x)
        x = self.add3([x, x_branch])
        x = self.relu3(x)

        x = self.global_avg_pool(x)
        x = self.dense(x)

        return x

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal(shape=(1, 32, 32, 3))
    output_tensor = model(input_tensor)
    print(output_tensor.shape)