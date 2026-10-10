import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Dropout, Add, Lambda, SeparableConv2D, Concatenate, Flatten, Dense
from tensorflow.keras.models import Model

class DL_Model(Model):
    def __init__(self):
        super(DL_Model, self).__init__()

        # First block - main path
        self.conv_expand = Conv2D(filters=64, kernel_size=1, strides=1, padding='same', activation='relu')
        self.dropout_expand = Dropout(rate=0.2)
        self.conv_restore = Conv2D(filters=32, kernel_size=1, strides=1, padding='same', activation='relu')

        # First block - branch path (direct connection)
        self.branch = Lambda(lambda x: x)

        # Second block - separable convolutions
        self.sep_conv1 = SeparableConv2D(filters=32, kernel_size=1, strides=1, padding='same', activation='relu')
        self.dropout1 = Dropout(rate=0.2)
        self.sep_conv3 = SeparableConv2D(filters=32, kernel_size=3, strides=1, padding='same', activation='relu')
        self.dropout3 = Dropout(rate=0.2)
        self.sep_conv5 = SeparableConv2D(filters=32, kernel_size=5, strides=1, padding='same', activation='relu')
        self.dropout5 = Dropout(rate=0.2)

        # Final layers
        self.flatten = Flatten()
        self.dense = Dense(units=10, activation='softmax')

    def call(self, inputs):
        # First block processing
        main_path = self.conv_expand(inputs)
        main_path = self.dropout_expand(main_path)
        main_path = self.conv_restore(main_path)

        # Add main path and branch path
        block1_output = Add()([main_path, self.branch(inputs)])

        # Second block processing
        # Split into three groups along channel dimension
        group1 = self.sep_conv1(block1_output)
        group1 = self.dropout1(group1)

        group3 = self.sep_conv3(block1_output)
        group3 = self.dropout3(group3)

        group5 = self.sep_conv5(block1_output)
        group5 = self.dropout5(group5)

        # Concatenate the groups
        block2_output = Concatenate(axis=-1)([group1, group3, group5])

        # Final prediction
        x = self.flatten(block2_output)
        output = self.dense(x)

        return output

if __name__ == "__main__":
    model = DL_Model()
    # CIFAR-10 images are 32x32 with 3 channels
    input_tensor = tf.keras.Input(shape=(32, 32, 3))
    output_tensor = model(input_tensor)
    print("Model summary:")
    model.summary()