import tensorflow as tf

class DL_Model(tf.keras.Model):
    def __init__(self):
        super(DL_Model, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(filters=16, kernel_size=3, strides=1, padding='same')
        self.bn1 = tf.keras.layers.BatchNormalization(axis=-1)
        self.relu1 = tf.keras.layers.ReLU()
        self.conv2_block = tf.keras.Sequential([
            tf.keras.layers.Conv2D(filters=16, kernel_size=3, strides=1, padding='same'),
            tf.keras.layers.BatchNormalization(axis=-1),
            tf.keras.layers.ReLU()
        ])
        self.conv3_block = tf.keras.Sequential([
            tf.keras.layers.Conv2D(filters=16, kernel_size=3, strides=1, padding='same'),
            tf.keras.layers.BatchNormalization(axis=-1),
            tf.keras.layers.ReLU()
        ])
        self.avg_pool = tf.keras.layers.AveragePooling2D(pool_size=(8, 8), strides=(1, 1), padding='valid')
        self.flatten = tf.keras.layers.Flatten()
        self.dense = tf.keras.layers.Dense(10, activation='softmax')

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.relu1(x)
        
        path1 = x
        
        branch = self.conv2_block(inputs)
        
        x = tf.keras.layers.Add()([path1, branch])
        x = self.relu1(x)
        
        x = self.conv3_block(x)
        
        x = self.avg_pool(x)
        x = self.flatten(x)
        x = self.dense(x)
        
        return x

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal((1, 32, 32, 3))
    output_tensor = model(input_tensor)
    print(output_tensor.shape)