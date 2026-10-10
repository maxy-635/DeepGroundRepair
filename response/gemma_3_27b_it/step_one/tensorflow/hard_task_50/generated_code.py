import tensorflow as tf

class DL_Model(tf.keras.Model):
    def __init__(self):
        super(DL_Model, self).__init__()

        # First block
        self.max_pool_1x1 = tf.keras.layers.MaxPool2D(pool_size=(1, 1), strides=(1, 1), padding='valid')
        self.max_pool_2x2 = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid')
        self.max_pool_4x4 = tf.keras.layers.MaxPool2D(pool_size=(4, 4), strides=(4, 4), padding='valid')
        self.flatten_1x1 = tf.keras.layers.Flatten()
        self.flatten_2x2 = tf.keras.layers.Flatten()
        self.flatten_4x4 = tf.keras.layers.Flatten()
        self.dropout = tf.keras.layers.Dropout(rate=0.5)
        self.dense = tf.keras.layers.Dense(units=64, activation='relu')
        
        # Second block
        self.separable_conv_1x1 = tf.keras.layers.DepthwiseConv2D(kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')
        self.separable_conv_3x3 = tf.keras.layers.DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')
        self.separable_conv_5x5 = tf.keras.layers.DepthwiseConv2D(kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')
        self.separable_conv_7x7 = tf.keras.layers.DepthwiseConv2D(kernel_size=(7, 7), strides=(1, 1), padding='same', activation='relu')
        self.concatenate = tf.keras.layers.Concatenate(axis=-1)
        self.flatten = tf.keras.layers.Flatten()
        self.output_dense = tf.keras.layers.Dense(units=10, activation='softmax')

    def call(self, inputs):
        # First block
        pool_1x1 = self.max_pool_1x1(inputs)
        pool_2x2 = self.max_pool_2x2(inputs)
        pool_4x4 = self.max_pool_4x4(inputs)

        flat_1x1 = self.flatten_1x1(pool_1x1)
        flat_2x2 = self.flatten_2x2(pool_2x2)
        flat_4x4 = self.flatten_4x4(pool_4x4)

        dropout_out = self.dropout(tf.concat([flat_1x1, flat_2x2, flat_4x4], axis=1))
        dense_out = self.dense(dropout_out)
        reshaped_out = tf.reshape(dense_out, [-1, 32, 32, 1]) #Assuming input size is 32x32

        # Second block
        group1 = self.separable_conv_1x1(reshaped_out)
        group2 = self.separable_conv_3x3(reshaped_out)
        group3 = self.separable_conv_5x5(reshaped_out)
        group4 = self.separable_conv_7x7(reshaped_out)

        concat_out = self.concatenate([group1, group2, group3, group4])
        flat_out = self.flatten(concat_out)
        output = self.output_dense(flat_out)
        
        return output

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal(shape=(1, 32, 32, 3))
    output_tensor = model(input_tensor)
    print(output_tensor.shape)