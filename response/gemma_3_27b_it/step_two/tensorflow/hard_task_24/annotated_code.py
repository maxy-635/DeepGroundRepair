import tensorflow as tf

class DL_Model(tf.keras.Model):
    def __init__(self):
        super(DL_Model, self).__init__()
        
        # Initial 1x1 Convolutional Layer
        self.conv1 = tf.keras.layers.Conv2D(filters=32, kernel_size=1, strides=1, padding='same', activation='relu')
        
        # Branch 1
        self.branch1_conv = tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=1, padding='same', activation='relu')
        
        # Branch 2
        self.branch2_pool = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2, padding='valid')
        self.branch2_conv = tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=1, padding='same', activation='relu')
        self.branch2_upsample = tf.keras.layers.UpSampling2D(size=(2, 2))
        
        # Branch 3
        self.branch3_pool = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2, padding='valid')
        self.branch3_conv = tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=1, padding='same', activation='relu')
        self.branch3_upsample = tf.keras.layers.UpSampling2D(size=(2, 2))
        
        # Concatenation and Fusion
        self.concatenate = tf.keras.layers.Concatenate(axis=-1)
        self.fusion_conv = tf.keras.layers.Conv2D(filters=64, kernel_size=1, strides=1, padding='same', activation='relu')
        
        # Fully Connected Layers
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(units=128, activation='relu')
        self.dense2 = tf.keras.layers.Dense(units=64, activation='relu')
        self.dense3 = tf.keras.layers.Dense(units=10, activation='softmax')
        
    def call(self, inputs):
        x = self.conv1(inputs)
        
        branch1 = self.branch1_conv(x)
        
        branch2 = self.branch2_pool(x)
        branch2 = self.branch2_conv(branch2)
        branch2 = self.branch2_upsample(branch2)
        
        branch3 = self.branch3_pool(x)
        branch3 = self.branch3_conv(branch3)
        branch3 = self.branch3_upsample(branch3)
        
        merged = self.concatenate([branch1, branch2, branch3])
        merged = self.fusion_conv(merged)
        
        flattened = self.flatten(merged)
        dense1 = self.dense1(flattened)
        dense2 = self.dense2(dense1)
        output = self.dense3(dense2)
        
        return output


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal(shape=(1, 32, 32, 3))
    output_tensor = model(input_tensor)
    print(output_tensor.shape)