import tensorflow as tf

class DL_Model(tf.keras.Model):
    def __init__(self):
        super(DL_Model, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(
            filters=32,
            kernel_size=7,
            strides=1,
            padding='same',
            activation='relu'
        )
        self.bn1 = tf.keras.layers.BatchNormalization(axis=-1)
        self.dense1 = tf.keras.layers.Dense(
            units=64,
            activation='relu'
        )
        self.dense2 = tf.keras.layers.Dense(
            units=10,
            activation='softmax'
        )

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.Flatten()(x)
        x = self.dense1(x)
        x = self.dense2(x)
        return x

if __name__ == "__main__":
    #(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    #x_train = x_train.astype('float32')
    #x_test = x_test.astype('float32')
    #x_train = x_train / 127.5 - 1.0
    #x_test = x_test / 127.5 - 1.0
    
    model = DL_Model()
    input_tensor = tf.keras.Input(shape=(32, 32, 3))
    output_tensor = model(input_tensor)
    print(output_tensor)