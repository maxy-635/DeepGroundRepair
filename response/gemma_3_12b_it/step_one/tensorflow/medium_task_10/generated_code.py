import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, ReLU, Add, AveragePooling2D, Dense, Input

class DL_Model(tf.keras.Model):
    def __init__(self):
        super(DL_Model, self).__init__()

        # First level: Basic block
        self.conv1_1 = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')
        self.bn1_1 = BatchNormalization(axis=-1)
        self.relu1_1 = ReLU()

        # Second level: Two residual blocks
        self.conv2_1 = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')
        self.bn2_1 = BatchNormalization(axis=-1)
        self.relu2_1 = ReLU()
        self.conv2_branch_1 = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same')
        self.bn2_branch_1 = BatchNormalization(axis=-1)

        self.conv2_2 = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')
        self.bn2_2 = BatchNormalization(axis=-1)
        self.relu2_2 = ReLU()
        self.conv2_branch_2 = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same')
        self.bn2_branch_2 = BatchNormalization(axis=-1)

        # Third level: Global branch
        self.conv3_global = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same')
        self.bn3_global = BatchNormalization(axis=-1)

        # Classification layers
        self.avg_pool = AveragePooling2D(pool_size=(8, 8), strides=(1, 1), padding='valid')
        self.dense = Dense(units=10, activation='softmax')

    def call(self, inputs):
        # First level
        x = self.conv1_1(inputs)
        x = self.bn1_1(x)
        x = self.relu1_1(x)
        identity = inputs

        x = Add()([x, identity])

        # Second level
        x2_1 = self.conv2_1(x)
        x2_1 = self.bn2_1(x2_1)
        x2_1 = self.relu2_1(x2_1)
        x2_branch_1 = self.conv2_branch_1(x)
        x2_branch_1 = self.bn2_branch_1(x2_branch_1)
        x2_1 = Add()([x2_1, x2_branch_1])

        x2_2 = self.conv2_2(x2_1)
        x2_2 = self.bn2_2(x2_2)
        x2_2 = self.relu2_2(x2_2)
        x2_branch_2 = self.conv2_branch_2(x2_1)
        x2_branch_2 = self.bn2_branch_2(x2_branch_2)
        x2_2 = Add()([x2_2, x2_branch_2])
        
        # Third level
        x3_global = self.conv3_global(inputs)
        x3_global = self.bn3_global(x3_global)

        x_final = Add()([x2_2, x3_global])

        # Classification
        x = self.avg_pool(x_final)
        x = self.dense(x)

        return x


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = Input(shape=(32, 32, 3))
    output_tensor = model(input_tensor)
    print(output_tensor)