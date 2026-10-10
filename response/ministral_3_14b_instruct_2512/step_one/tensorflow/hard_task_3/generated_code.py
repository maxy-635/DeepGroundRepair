import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Concatenate, Add, Lambda, Flatten, Activation, BatchNormalization
from tensorflow.keras.models import Model

class DL_Model(Model):
    def __init__(self):
        super(DL_Model, self).__init__()
        # Main pathway components
        self.conv1x1_1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')
        self.conv3x3_1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')
        self.dropout_1 = Dropout(rate=0.2)

        self.conv1x1_2 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')
        self.conv3x3_2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')
        self.dropout_2 = Dropout(rate=0.2)

        self.conv1x1_3 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')
        self.conv3x3_3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')
        self.dropout_3 = Dropout(rate=0.2)

        # Concatenate the three groups
        self.concat = Concatenate(axis=-1)

        # Branch pathway
        self.conv1x1_branch = Conv2D(filters=192, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')

        # Final layers
        self.add = Add()
        self.flatten = Flatten()
        self.dense = Dense(units=10, activation='softmax')

    def call(self, inputs):
        # Split input into three groups along channel dimension
        input_shape = tf.shape(inputs)
        batch_size = input_shape[0]
        height = input_shape[1]
        width = input_shape[2]
        channels = input_shape[3]

        # Split into three equal parts (assuming channels is divisible by 3)
        channels_per_group = channels // 3
        group1 = inputs[:, :, :, :channels_per_group]
        group2 = inputs[:, :, :, channels_per_group:2*channels_per_group]
        group3 = inputs[:, :, :, 2*channels_per_group:]

        # Process each group through the main pathway
        x1 = self.conv1x1_1(group1)
        x1 = self.conv3x3_1(x1)
        x1 = self.dropout_1(x1)

        x2 = self.conv1x1_2(group2)
        x2 = self.conv3x3_2(x2)
        x2 = self.dropout_2(x2)

        x3 = self.conv1x1_3(group3)
        x3 = self.conv3x3_3(x3)
        x3 = self.dropout_3(x3)

        # Concatenate the three groups
        main_pathway = self.concat([x1, x2, x3])

        # Process through branch pathway
        branch_pathway = self.conv1x1_branch(inputs)

        # Combine pathways
        combined = self.add([main_pathway, branch_pathway])

        # Final classification
        output = self.flatten(combined)
        output = self.dense(output)

        return output

if __name__ == "__main__":
    # Create model instance
    model = DL_Model()

    # Create dummy input tensor with shape (batch_size, height, width, channels)
    # CIFAR-10 images are 32x32 with 3 channels
    input_tensor = tf.random.normal(shape=(32, 32, 32, 3))

    # Get model output
    output_tensor = model(input_tensor)
    print("Input shape:", input_tensor.shape)
    print("Output shape:", output_tensor.shape)