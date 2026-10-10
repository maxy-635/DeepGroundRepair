import tensorflow as tf

class DL_Model(tf.keras.Model):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Branch 1: 3x3 Convolution
        self.conv3x3_1 = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')
        self.conv3x3_2 = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')

        # Branch 2: 5x5 Convolution
        self.conv5x5_1 = tf.keras.layers.Conv2D(filters=32, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')
        self.conv5x5_2 = tf.keras.layers.Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')

        # Combine branches
        self.add_layer = tf.keras.layers.Add()

        # Global Average Pooling
        self.global_avg_pool = tf.keras.layers.GlobalAveragePooling2D(keepdims=False)

        # Attention weights
        self.dense_attention_1 = tf.keras.layers.Dense(units=64, activation='relu')
        self.softmax = tf.keras.layers.Softmax(axis=-1)

        # Weighted output
        self.multiply_branch1 = tf.keras.layers.Multiply()
        self.multiply_branch2 = tf.keras.layers.Multiply()
        self.add_weighted = tf.keras.layers.Add()

        # Final classification layer
        self.dense_output = tf.keras.layers.Dense(units=10, activation='softmax')

    def call(self, inputs):
        # Branch 1
        branch1 = self.conv3x3_1(inputs)
        branch1 = self.conv3x3_2(branch1)

        # Branch 2
        branch2 = self.conv5x5_1(inputs)
        branch2 = self.conv5x5_2(branch2)

        # Combine branches
        combined = self.add_layer([branch1, branch2])

        # Global Average Pooling
        pooled = self.global_avg_pool(combined)

        # Attention weights
        attention = self.dense_attention_1(pooled)
        attention = self.softmax(attention)

        # Weighted output
        weighted_branch1 = self.multiply_branch1([branch1, attention])
        weighted_branch2 = self.multiply_branch2([branch2, attention])
        weighted_output = self.add_weighted([weighted_branch1, weighted_branch2])
        
        # Final classification
        output = self.dense_output(weighted_output)
        return output


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal(shape=(1, 32, 32, 3))  # Dummy input tensor with shape (batch_size, height, width, channels)
    output_tensor = model(input_tensor)
    print(output_tensor.shape)