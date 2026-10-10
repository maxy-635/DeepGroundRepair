import tensorflow as tf

class DL_Model(tf.keras.Model):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Input image channels
        input_channels = 3

        # Block 1: Channel Attention
        path1_gap = tf.keras.layers.GlobalAveragePooling2D(keepdims=True)
        path1_dense1 = tf.keras.layers.Dense(units=input_channels, activation='relu')
        path1_dense2 = tf.keras.layers.Dense(units=input_channels, activation='sigmoid')

        path2_gmp = tf.keras.layers.GlobalMaxPool2D(keepdims=True)
        path2_dense1 = tf.keras.layers.Dense(units=input_channels, activation='relu')
        path2_dense2 = tf.keras.layers.Dense(units=input_channels, activation='sigmoid')

        self.path1 = tf.keras.Sequential([path1_gap, path1_dense1, path1_dense2])
        self.path2 = tf.keras.Sequential([path2_gmp, path2_dense1, path2_dense2])

        self.channel_attention = tf.keras.layers.Multiply()
        self.channel_activation = tf.keras.layers.Activation('sigmoid')


        # Block 2: Spatial Attention
        self.avg_pool = tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')
        self.max_pool = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')
        self.concat = tf.keras.layers.Concatenate(axis=-1)
        self.conv1x1 = tf.keras.layers.Conv2D(filters=1, kernel_size=(1, 1), strides=(1, 1), padding='same')
        self.spatial_activation = tf.keras.layers.Activation('sigmoid')
        self.spatial_multiply = tf.keras.layers.Multiply()

        # Additional Branch
        self.conv1x1_branch = tf.keras.layers.Conv2D(filters=input_channels, kernel_size=(1, 1), strides=(1, 1), padding='same')
        self.branch_activation = tf.keras.layers.Activation('relu')
        self.add_layer = tf.keras.layers.Add()

        # Classification
        self.flatten = tf.keras.layers.Flatten()
        self.dense_classifier = tf.keras.layers.Dense(units=10, activation='softmax')

    def call(self, inputs):
        # Block 1
        path1_output = self.path1(inputs)
        path2_output = self.path2(inputs)
        attention_weights = self.channel_attention([path1_output, path2_output])
        attention_weights = self.channel_activation(attention_weights)
        channel_aware_features = self.spatial_multiply([inputs, attention_weights])

        # Block 2
        avg_pool_output = self.avg_pool(channel_aware_features)
        max_pool_output = self.max_pool(channel_aware_features)
        concat_output = self.concat([avg_pool_output, max_pool_output])
        spatial_features = self.conv1x1(concat_output)
        spatial_features = self.spatial_activation(spatial_features)
        spatial_aware_features = self.spatial_multiply([channel_aware_features, spatial_features])

        # Additional Branch
        branch_output = self.conv1x1_branch(inputs)
        branch_output = self.branch_activation(branch_output)
        added_features = self.add_layer([spatial_aware_features, branch_output])
        added_features = self.branch_activation(added_features)

        # Classification
        flattened_features = self.flatten(added_features)
        output = self.dense_classifier(flattened_features)

        return output

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal(shape=(1, 32, 32, 3))  # CIFAR-10 input shape
    output_tensor = model(input_tensor)
    print(output_tensor.shape) # Expected output shape: (1, 10)