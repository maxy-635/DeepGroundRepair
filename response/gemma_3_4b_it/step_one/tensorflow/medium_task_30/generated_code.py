import tensorflow as tf

class DL_Model(tf.keras.Model):
    def __init__(self):
        super(DL_Model, self).__init__()
        self.avg_pool1 = tf.keras.layers.GlobalAveragePooling2D(keepdims=True)
        self.avg_pool2 = tf.keras.layers.GlobalAveragePooling2D(keepdims=True)
        self.avg_pool3 = tf.keras.layers.GlobalAveragePooling2D(keepdims=True)
        self.flatten1 = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(units=128, activation='relu')
        self.dense2 = tf.keras.layers.Dense(units=10, activation='softmax')

    def call(self, inputs):
        x = self.avg_pool1(inputs)
        x = self.avg_pool2(x)
        x = self.avg_pool3(x)
        x = self.flatten1(x)
        x = self.dense1(x)
        x = self.dense2(x)
        return x

if __name__ == "__main__":
    #(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    #x_train = x_train / 255.0
    #x_test = x_test / 255.0

    model = DL_Model()
    input_tensor = tf.random.normal((1, 32, 32, 3))  # Batch size 1, 32x32 image, 3 channels
    output_tensor = model(input_tensor)
    print(output_tensor.shape)