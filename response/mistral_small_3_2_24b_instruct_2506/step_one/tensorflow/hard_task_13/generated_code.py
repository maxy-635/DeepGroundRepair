import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPool2D, Concatenate, GlobalAveragePooling2D, Dense, Multiply, Reshape, Activation
from tensorflow.keras.models import Model

class DL_Model(Model):
    def __init__(self):
        super(DL_Model, self).__init__()
        # First block: parallel branches
        self.conv1x1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')
        self.conv3x3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')
        self.conv5x5 = Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')
        self.max_pool = MaxPool2D(pool_size=(3, 3), strides=(1, 1), padding='same')
        self.concat = Concatenate(axis=-1)

        # Second block: dimensionality reduction and attention
        self.global_avg_pool = GlobalAveragePooling2D(keepdims=False)
        self.dense1 = Dense(units=64, activation='relu')
        self.dense2 = Dense(units=64, activation='sigmoid')
        self.reshape = Reshape(target_shape=(-1, 64, 1, 1))
        self.multiply = Multiply()
        self.dense_final = Dense(units=10, activation='softmax')

    def call(self, inputs):
        # First block: parallel branches
        conv1x1_out = self.conv1x1(inputs)
        conv3x3_out = self.conv3x3(inputs)
        conv5x5_out = self.conv5x5(inputs)
        max_pool_out = self.max_pool(inputs)
        concat_out = self.concat([conv1x1_out, conv3x3_out, conv5x5_out, max_pool_out])

        # Second block: dimensionality reduction and attention
        pooled = self.global_avg_pool(concat_out)
        dense1_out = self.dense1(pooled)
        dense2_out = self.dense2(dense1_out)
        reshaped = self.reshape(dense2_out)
        multiplied = self.multiply([concat_out, reshaped])
        output = self.dense_final(multiplied)

        return output

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal(shape=(1, 32, 32, 3))  # CIFAR-10 input shape
    output_tensor = model(input_tensor)
    print("Model output shape:", output_tensor.shape)