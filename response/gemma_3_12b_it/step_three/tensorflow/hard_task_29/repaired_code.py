import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Concatenate, Dense, Add, Input

class DL_Model(tf.keras.Model):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Block 1
        self.input1 = Input(shape=(28, 28, 1), dtype='float32', name='input_1')
        self.conv1_main = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', name='conv1_main')
        self.conv2_main = Conv2D(filters=1, kernel_size=(3, 3), strides=(1, 1), padding='same', activation=None, name='conv2_main')
        self.conv1_branch = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', name='conv1_branch')
        self.add1 = Add(name='add1')

        # Block 2
        self.maxpool1 = MaxPool2D(pool_size=(1, 1), strides=(1, 1), padding='valid', name='maxpool1')
        self.maxpool2 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid', name='maxpool2')
        self.maxpool3 = MaxPool2D(pool_size=(4, 4), strides=(4, 4), padding='valid', name='maxpool3')
        self.flatten1 = Flatten(name='flatten1')
        self.concatenate = Concatenate(axis=-1, name='concatenate')

        # Classifier
        self.dense1 = Dense(units=128, activation='relu', name='dense1')
        self.dense2 = Dense(units=10, activation='softmax', name='dense2')

    def call(self, inputs):
        # Block 1
        main_path = self.conv1_main(inputs)
        main_path = self.conv2_main(main_path)
        branch_path = self.conv1_branch(inputs)
        combined_output = self.add1([main_path, branch_path])

        # Block 2
        pooled1 = self.maxpool1(combined_output)
        pooled2 = self.maxpool2(pooled1)
        pooled3 = self.maxpool3(pooled2)
        flattened = self.flatten1(pooled3)
        concatenated = self.concatenate([self.flatten1(self.maxpool1(combined_output)),
                                         self.flatten1(self.maxpool2(combined_output)),
                                         self.flatten1(self.maxpool3(combined_output))])

        # Classifier
        dense1_output = self.dense1(concatenated)
        output = self.dense2(dense1_output)

        return output


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal((1, 28, 28, 1))  # Dummy input tensor
    output_tensor = model(input_tensor)
    print("Output shape:", output_tensor.shape)