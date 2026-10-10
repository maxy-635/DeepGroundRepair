import tensorflow as tf
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Reshape, Multiply, Conv2D, MaxPool2D, Add, Flatten, concatenate
from tensorflow.keras.models import Model

class DL_Model(Model):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Block 1: Attention mechanism
        self.global_avg_pool = GlobalAveragePooling2D(keepdims='mask')
        self.dense1 = Dense(units=32, activation='relu')
        self.dense2 = Dense(units='mask', activation='sigmoid')
        self.reshape = Reshape(target_shape=(1, 1, 32))
        self.multiply = Multiply()

        # Block 2: Feature extraction
        self.conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')
        self.conv2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')
        self.max_pool = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')

        # Output layers
        self.flatten = Flatten()
        self.fc1 = Dense(units=128, activation='relu')
        self.fc2 = Dense(units=10, activation='softmax')

    def call(self, inputs):
        # the initial shape of inputs is: (1, 32, 32, 3)
        # Block 1: Attention mechanism
        attention_weights = self.global_avg_pool(inputs)  # output tensor shape of this layer should be (1, 32, 32, 3)
        attention_weights = self.dense1(attention_weights)
        attention_weights = self.dense2(attention_weights)  # output tensor shape of this layer should be (1, 1, 1, 3)
        attention_weights = self.reshape(attention_weights)
        weighted_features = self.multiply([inputs, attention_weights])

        # Block 2: Feature extraction
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.max_pool(x)

        # Branch connection
        branch_output = weighted_features

        # Fusion
        fused_output = Add()([x, branch_output])

        # Output layers
        x = self.flatten(fused_output)
        x = self.fc1(x)
        outputs = self.fc2(x)

        return outputs

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal(shape=(1, 32, 32, 3))  # CIFAR-10 image shape
    output_tensor = model(input_tensor)
    print("Input shape:", input_tensor.shape)
    print("Output shape:", output_tensor.shape)