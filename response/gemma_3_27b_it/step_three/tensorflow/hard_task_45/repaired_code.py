import tensorflow as tf

class DL_Model(tf.keras.Model):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Block 1
        self.depthwise_1x1 = tf.keras.layers.DepthwiseConv2D(
            kernel_size=(1, 1),
            strides=(1, 1),
            padding='same',
            depth_multiplier=1,
            activation='relu'
        )
        self.depthwise_3x3 = tf.keras.layers.DepthwiseConv2D(
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='same',
            depth_multiplier=1,
            activation='relu'
        )
        self.depthwise_5x5 = tf.keras.layers.DepthwiseConv2D(
            kernel_size=(5, 5),
            strides=(1, 1),
            padding='same',
            depth_multiplier=1,
            activation='relu'
        )
        self.concatenate_1 = tf.keras.layers.Concatenate(axis=-1)

        # Block 2 - Branches
        self.branch_1 = tf.keras.layers.Conv2D(
            filters=32,
            kernel_size=(1, 1),
            strides=(1, 1),
            padding='same',
            activation='relu'
        )

        self.branch_2_conv1 = tf.keras.layers.Conv2D(
            filters=32,
            kernel_size=(1, 1),
            strides=(1, 1),
            padding='same',
            activation='relu'
        )
        self.branch_2_conv2 = tf.keras.layers.Conv2D(
            filters=32,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='same',
            activation='relu'
        )
        self.branch_2_conv3 = tf.keras.layers.Conv2D(
            filters=32,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='same',
            activation='relu'
        )
        self.concatenate_2 = tf.keras.layers.Concatenate(axis=-1)

        self.branch_3_conv1 = tf.keras.layers.Conv2D(
            filters=32,
            kernel_size=(1, 1),
            strides=(1, 1),
            padding='same',
            activation='relu'
        )
        self.branch_3_conv2 = tf.keras.layers.Conv2D(
            filters=32,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='same',
            activation='relu'
        )
        self.concatenate_3 = tf.keras.layers.Concatenate(axis=-1)

        self.branch_4_pool = tf.keras.layers.MaxPool2D(pool_size=(1, 1), strides=(1, 1), padding='same')




        self.branch_4_conv = tf.keras.layers.Conv2D(
            filters=32,
            kernel_size=(1, 1),
            strides=(1, 1),
            padding='same',
            activation='relu'
        )
        self.concatenate_4 = tf.keras.layers.Concatenate(axis=-1)

        self.concatenate_all = tf.keras.layers.Concatenate(axis=-1)

        # Final Layers
        self.flatten = tf.keras.layers.Flatten()
        self.dense = tf.keras.layers.Dense(units=10, activation='softmax')

    def call(self, inputs):
        # the initial shape of inputs is: (1, 32, 32, 3)
        # Block 1
        x1 = self.depthwise_1x1(inputs)  # output tensor shape: (1, 32, 32, 3)
        x2 = self.depthwise_3x3(inputs)  # output tensor shape: (1, 32, 32, 3)
        x3 = self.depthwise_5x5(inputs)  # output tensor shape: (1, 32, 32, 3)
        x = self.concatenate_1([x1, x2, x3])  # output tensor shape: (1, 32, 32, 9)

        # Block 2
        b1 = self.branch_1(x)  # output tensor shape: (1, 32, 32, 32)

        b2_1 = self.branch_2_conv1(x)  # output tensor shape: (1, 32, 32, 32)
        b2_2 = self.branch_2_conv2(b2_1)  # output tensor shape: (1, 32, 32, 32)
        b2_3 = self.branch_2_conv3(b2_2)  # output tensor shape: (1, 32, 32, 32)
        b2 = self.concatenate_2([b2_1, b2_2, b2_3])  # output tensor shape: (1, 32, 32, 96)
        
        b3_1 = self.branch_3_conv1(x)  # output tensor shape: (1, 32, 32, 32)
        b3_2 = self.branch_3_conv2(b3_1)  # output tensor shape: (1, 32, 32, 32)
        b3 = self.concatenate_3([b3_1, b3_2])  # output tensor shape: (1, 32, 32, 64)

        b4_pool = self.branch_4_pool(x)  # output tensor shape of this layer should be (1, 32, 32, 9)
        b4_conv = self.branch_4_conv(b4_pool)
        b4 = self.concatenate_4([b4_pool, b4_conv])

        x = self.concatenate_all([b1, b2, b3, b4])

        # Final Layers
        x = self.flatten(x)
        return self.dense(x)


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal((1, 32, 32, 3))
    output_tensor = model(input_tensor)
    print(output_tensor.shape)