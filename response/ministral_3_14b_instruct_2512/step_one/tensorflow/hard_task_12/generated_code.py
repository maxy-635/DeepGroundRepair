import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Conv1D, Dense, Flatten, Concatenate, Add, Lambda
from tensorflow.keras.models import Model

class DL_Model(Model):
    def __init__(self):
        super(DL_Model, self).__init__()
        # Main path
        self.conv1x1_main = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')
        self.conv1x1_feat = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')
        self.conv3x3_feat = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')
        self.concat = Concatenate(axis=-1)

        # Branch path
        self.conv3x3_branch = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')

        # Final layers
        self.flatten = Flatten()
        self.dense1 = Dense(units=128, activation='relu')
        self.dense2 = Dense(units=10, activation='softmax')  # Assuming 10 classes for classification

    def call(self, inputs):
        # Main path
        x = self.conv1x1_main(inputs)
        feat1 = self.conv1x1_feat(x)
        feat2 = self.conv3x3_feat(x)
        main_output = self.concat([feat1, feat2])

        # Branch path
        branch_output = self.conv3x3_branch(inputs)

        # Combine main and branch outputs
        combined = Add()([main_output, branch_output])

        # Final layers
        x = self.flatten(combined)
        x = self.dense1(x)
        x = self.dense2(x)

        return x

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal(shape=(1, 32, 32, 64))  # Batch size of 1, shape (32, 32, 64)
    output_tensor = model(input_tensor)
    print("Output shape:", output_tensor.shape)