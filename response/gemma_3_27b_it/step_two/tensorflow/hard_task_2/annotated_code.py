import tensorflow as tf

class DL_Model(tf.keras.Model):
    def __init__(self):
        super(DL_Model, self).__init__()
        
        # Define the layers
        self.conv1_1 = tf.keras.layers.Conv2D(filters=32, kernel_size=1, strides=1, padding='same', activation='relu')
        self.conv1_3 = tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=1, padding='same', activation='relu')
        self.conv1_1_2 = tf.keras.layers.Conv2D(filters=32, kernel_size=1, strides=1, padding='same', activation='relu')
        
        self.conv2_1 = tf.keras.layers.Conv2D(filters=32, kernel_size=1, strides=1, padding='same', activation='relu')
        self.conv2_3 = tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=1, padding='same', activation='relu')
        self.conv2_1_2 = tf.keras.layers.Conv2D(filters=32, kernel_size=1, strides=1, padding='same', activation='relu')
        
        self.conv3_1 = tf.keras.layers.Conv2D(filters=32, kernel_size=1, strides=1, padding='same', activation='relu')
        self.conv3_3 = tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=1, padding='same', activation='relu')
        self.conv3_1_2 = tf.keras.layers.Conv2D(filters='mask', kernel_size=1, strides=1, padding='same', activation='relu')
        
        self.flatten = tf.keras.layers.Flatten()
        self.dense = tf.keras.layers.Dense(units=10, activation='softmax')

    def call(self, inputs):
        # the initial shape of inputs is: (1, 32, 32, 3)
        # Split input into three groups along the channel dimension
        channel_dim = inputs.shape[-1]  # output tensor shape: (1)
        group1 = inputs[:, :, :, :channel_dim//3]  # output tensor shape: (1, 32, 32, 1)
        group2 = inputs[:, :, :, channel_dim//3:2*channel_dim//3]  # output tensor shape: (1, 32, 32, 1)
        group3 = inputs[:, :, :, 2*channel_dim//3:]  # output tensor shape: (1, 32, 32, 1)
        
        # Process each group through convolutions
        path1 = self.conv1_1(group1)  # output tensor shape: (1, 32, 32, 32)
        path1 = self.conv1_3(path1)  # output tensor shape: (1, 32, 32, 32)
        path1 = self.conv1_1_2(path1)  # output tensor shape: (1, 32, 32, 32)
        
        path2 = self.conv2_1(group2)  # output tensor shape: (1, 32, 32, 32)
        path2 = self.conv2_3(path2)  # output tensor shape: (1, 32, 32, 32)
        path2 = self.conv2_1_2(path2)  # output tensor shape: (1, 32, 32, 32)
        
        path3 = self.conv3_1(group3)  # output tensor shape: (1, 32, 32, 32)
        path3 = self.conv3_3(path3)  # output tensor shape: (1, 32, 32, 32)
        path3 = self.conv3_1_2(path3)  # output tensor shape of this layer should be (1, 32, 32, 3)
        
        # Combine the outputs
        main_path = tf.keras.layers.Add()([path1, path2, path3])
        
        # Fuse with the original input
        combined_features = tf.keras.layers.Add()([main_path, inputs])
        
        # Flatten and pass through a fully connected layer
        flattened = self.flatten(combined_features)
        output = self.dense(flattened)
        
        return output


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal((1, 32, 32, 3))  # Example input tensor with shape (batch_size, height, width, channels)
    output_tensor = model(input_tensor)
    print(output_tensor.shape)