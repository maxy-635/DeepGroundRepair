import tensorflow as tf

class DL_Model(tf.keras.Model):
    def __init__(self):
        super(DL_Model, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(
            filters=32,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='same',
            activation='relu'
        )
        self.conv2_1 = tf.keras.layers.Conv2D(
            filters=64,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='same',
            activation='relu'
        )
        self.conv2_2 = tf.keras.layers.Conv2D(
            filters=64,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='same',
            activation='relu'
        )
        self.max_pool = tf.keras.layers.MaxPool2D(
            pool_size=(2, 2),
            strides=(2, 2),
            padding='same'
        )
        self.dropout = tf.keras.layers.Dropout(
            rate=0.5
        )
        self.conv_branch = tf.keras.layers.Conv2D(
            filters=64,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='same',
            activation='relu'
        )
        self.global_avg_pool = tf.keras.layers.GlobalAveragePooling2D(keepdims=True)
        self.flatten = tf.keras.layers.Flatten()
        self.dense = tf.keras.layers.Dense(
            units=10,
            activation='softmax'
        )

    def call(self, inputs):
        main_path = self.conv1(inputs)
        main_path = self.conv2_1(main_path)
        main_path = self.conv2_2(main_path)
        main_path = self.max_pool(main_path)
        main_path = self.dropout(main_path)

        branch_path = self.conv_branch(inputs)
        merged = tf.keras.layers.Add()([main_path, branch_path])
        merged = self.global_avg_pool(merged)
        merged = self.flatten(merged)
        output = self.dense(merged)
        return output

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal((1, 28, 28, 1))
    output_tensor = model(input_tensor)
    print(output_tensor.shape)