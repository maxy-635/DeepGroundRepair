import tensorflow as tf
from tensorflow.keras.layers import Layer, Conv2D, BatchNormalization, Add, AveragePooling2D, Dense
from tensorflow.keras.models import Model

class BasicBlock(Layer):
    def __init__(self, filters, **kwargs):
        super(BasicBlock, self).__init__(**kwargs)
        self.conv1 = Conv2D(filters=filters, kernel_size=3, padding='same')
        self.bn1 = BatchNormalization(axis=-1)
        self.conv2 = Conv2D(filters=filters, kernel_size=3, padding='same')
        self.bn2 = BatchNormalization(axis=-1)

    def call(self, inputs, training=False):
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = tf.nn.relu(x)
        x = self.conv2(x)
        x = self.bn2(x, training=training)
        x = tf.math.add(x, inputs)
        return tf.nn.relu(x)

class ResidualBlock(Layer):
    def __init__(self, filters, **kwargs):
        super(ResidualBlock, self).__init__(**kwargs)
        self.basic_block1 = BasicBlock(filters=filters)
        self.basic_block2 = BasicBlock(filters=filters)
        self.conv = Conv2D(filters=filters, kernel_size=1, strides=1)

    def call(self, inputs, training=False):
        x = self.basic_block1(inputs, training=training)
        x = self.basic_block2(x, training=training)
        residual = self.conv(inputs)
        return tf.math.add(x, residual)

class DL_Model(Model):
    def __init__(self, **kwargs):
        super(DL_Model, self).__init__(**kwargs)
        self.conv1 = Conv2D(filters=16, kernel_size=3, padding='same')
        self.basic_block = BasicBlock(filters=16)
        self.residual_block1 = ResidualBlock(filters=32)
        self.residual_block2 = ResidualBlock(filters=64)
        self.global_conv = Conv2D(filters=64, kernel_size=1, strides=1)
        self.avg_pool = AveragePooling2D(pool_size=4, strides=1, padding='valid')
        self.dense = Dense(units=10, activation='softmax')

    def call(self, inputs, training=False):
        x = self.conv1(inputs)
        x = self.basic_block(x, training=training)
        x = self.residual_block1(x, training=training)
        x = self.residual_block2(x, training=training)
        global_features = self.global_conv(x)
        x = tf.math.add(x, global_features)
        x = self.avg_pool(x)
        x = tf.keras.layers.Flatten()(x)
        return self.dense(x)

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.keras.Input(shape=(32, 32, 3))
    output_tensor = model(input_tensor)
    print(output_tensor.shape)