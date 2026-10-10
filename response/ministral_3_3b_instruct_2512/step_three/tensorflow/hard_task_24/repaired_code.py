import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPool2D, UpSampling2D, Concatenate, Dense, Flatten, GlobalAveragePooling2D, BatchNormalization, Dropout, LeakyReLU

class DL_Model(tf.keras.Model):
    def __init__(self):
        super(DL_Model, self).__init__()
        self.conv1x1 = Conv2D(64, kernel_size=(1, 1), strides=(1, 1), padding='same')
        self.conv3x3_branch1 = Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding='same')
        self.maxpool1 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid')
        self.conv3x3_branch2 = Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding='same')
        self.upsample1 = UpSampling2D(size=(2, 2))
        self.conv3x3_branch3 = Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding='same')
        self.upsample2 = UpSampling2D(size=(2, 2))
        self.concatenate = Concatenate(axis=-1)
        self.conv1x1_final = Conv2D(10, kernel_size=(1, 1), strides=(1, 1), padding='same')
        self.dense1 = Dense(512, activation='relu')
        self.dense2 = Dense(256, activation='relu')
        self.dense3 = Dense(10, activation='softmax')

    def call(self, inputs):
        x = self.conv1x1(inputs)
        branch1 = self.conv3x3_branch1(x)
        branch2 = self.maxpool1(self.conv3x3_branch2(x))
        branch2 = self.upsample1(branch2)
        branch3 = self.maxpool1(self.conv3x3_branch3(x))
        branch3 = self.upsample2(branch3)
        fused = self.concatenate([branch1, branch2, branch3])
        x = self.conv1x1_final(fused)
        x = Flatten()(x)
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.dense3(x)
        return x

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal((1, 32, 32, 3))
    output_tensor = model(input_tensor)