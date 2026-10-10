import tensorflow as tf

class DL_Model(tf.keras.Model):
    def __init__(self):
        super(DL_Model, self).__init__()
        self.block1 = tf.keras.Sequential([
            tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=1, padding='same', activation='relu'),
            tf.keras.layers.Conv2D(filters=32, kernel_size=1, strides=1, padding='same', activation='relu'),
            tf.keras.layers.Conv2D(filters=32, kernel_size=1, strides=1, padding='same', activation='relu'),
            tf.keras.layers.AveragePooling2D(pool_size=2, strides=2, padding='same'),
            tf.keras.layers.Dropout(rate=0.25)
        ])

        self.block2 = tf.keras.Sequential([
            tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=1, padding='same', activation='relu'),
            tf.keras.layers.Conv2D(filters=64, kernel_size=1, strides=1, padding='same', activation='relu'),
            tf.keras.layers.Conv2D(filters=64, kernel_size=1, strides=1, padding='same', activation='relu'),
            tf.keras.layers.AveragePooling2D(pool_size=2, strides=2, padding='same'),
            tf.keras.layers.Dropout(rate=0.25)
        ])
        
        self.global_avg_pool = tf.keras.layers.GlobalAveragePooling2D(keepdims=False)
        self.flatten = tf.keras.layers.Flatten()
        self.dense = tf.keras.layers.Dense(units=10, activation='softmax')

    def call(self, inputs):
        x = self.block1(inputs)
        x = self.block2(x)
        x = self.global_avg_pool(x)
        x = self.flatten(x)
        return self.dense(x)


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal((1, 28, 28, 1))
    output_tensor = model(input_tensor)
    print(output_tensor.shape)