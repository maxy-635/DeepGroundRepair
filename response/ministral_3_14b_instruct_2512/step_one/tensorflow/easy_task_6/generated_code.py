import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Add, Flatten, Dense, Lambda, Concatenate, Reshape
from tensorflow.keras.models import Model

class DL_Model(Model):
    def __init__(self):
        super(DL_Model, self).__init__()
        # Main path layers
        self.conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')
        self.conv2 = Conv2D(filters=1, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')

        # Branch path (identity connection)
        self.branch = Lambda(lambda x: x)

    def call(self, inputs):
        # Main path
        x = self.conv1(inputs)
        x = self.conv2(x)

        # Branch path (identity)
        branch = self.branch(inputs)

        # Combine paths through addition
        combined = Add()([x, branch])

        # Flatten and dense layers
        x = Flatten()(combined)
        x = Dense(units=10, activation='softmax')(x)

        return x

if __name__ == "__main__":
    model = DL_Model()
    # MNIST images are 28x28 with 1 channel
    input_tensor = tf.random.normal(shape=(1, 28, 28, 1))
    output_tensor = model(input_tensor)
    print("Output shape:", output_tensor.shape)