import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Dropout, Concatenate, Add, Dense, Lambda
from tensorflow.keras.models import Model

class DL_Model(Model):
    def __init__(self):
        super(DL_Model, self).__init__()
        # Split the input into three groups along the channel dimension
        self.split = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))

        # Define the three branches with 1x1 and 3x3 convolutions
        self.branch1_conv1x1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')
        self.branch1_conv3x3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')
        self.branch1_dropout = Dropout(rate=0.5)

        self.branch2_conv1x1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')
        self.branch2_conv3x3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')
        self.branch2_dropout = Dropout(rate=0.5)

        self.branch3_conv1x1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')
        self.branch3_conv3x3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')
        self.branch3_dropout = Dropout(rate=0.5)

        # Concatenate the outputs of the three branches
        self.concatenate = Concatenate(axis=-1)

        # Define the branch pathway
        self.branch_pathway = Conv2D(filters=192, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')

        # Define the addition operation
        self.add = Add()

        # Define the fully connected layer
        self.dense = Dense(units=10, activation='softmax')

    def call(self, inputs):
        # Split the input into three groups
        split_inputs = self.split(inputs)

        # Process each branch
        branch1 = self.branch1_conv1x1(split_inputs[0])
        branch1 = self.branch1_conv3x3(branch1)
        branch1 = self.branch1_dropout(branch1)

        branch2 = self.branch2_conv1x1(split_inputs[1])
        branch2 = self.branch2_conv3x3(branch2)
        branch2 = self.branch2_dropout(branch2)

        branch3 = self.branch3_conv1x1(split_inputs[2])
        branch3 = self.branch3_conv3x3(branch3)
        branch3 = self.branch3_dropout(branch3)

        # Concatenate the outputs of the three branches
        main_pathway = self.concatenate([branch1, branch2, branch3])

        # Process the branch pathway
        branch_pathway = self.branch_pathway(inputs)

        # Combine the outputs using addition
        combined = self.add([main_pathway, branch_pathway])

        # Pass through the fully connected layer
        output = self.dense(tf.keras.layers.GlobalAveragePooling2D()(combined))

        return output

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.keras.Input(shape=(32, 32, 3))
    output_tensor = model(input_tensor)
    print(output_tensor)