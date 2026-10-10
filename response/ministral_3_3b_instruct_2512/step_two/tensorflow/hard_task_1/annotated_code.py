import tensorflow as tf
from tensorflow.keras.layers import Conv2D, GlobalAveragePooling2D, GlobalMaxPool2D, Dense, Add, Multiply, Activation, concatenate
from tensorflow.keras.ops import sigmoid

class DL_Model(tf.keras.Model):
    def __init__(self):
        super(DL_Model, self).__init__()
        self.conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')
        self.pool1 = GlobalAveragePooling2D(keepdims=True)
        self.pool2 = GlobalMaxPool2D(keepdims=True)
        self.fc1 = Dense(units=64, activation='relu')
        self.fc2 = Dense(units=32, activation='relu')
        self.conv2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')
        self.conv3 = Conv2DTranspose(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')

    def call(self, inputs):
        x = self.conv1(inputs)
        path1 = self.pool1(x)
        path1 = self.fc1(path1)
        path1 = self.fc2(path1)

        path2 = self.pool2(x)
        path2 = self.fc1(path2)
        path2 = self.fc2(path2)

        attention_weights = Add()([path1, path2])
        attention_weights = Activation('sigmoid')(attention_weights)
        x = Multiply()([x, attention_weights])

        spatial_features = concatenate([self.pool1(x), self.pool2(x)], axis=-1)
        spatial_features = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='same')(spatial_features)
        spatial_features = Activation('sigmoid')(spatial_features)
        x = Multiply()([x, spatial_features])

        x = Add()([x, self.conv3(x)])
        x = Activation('relu')(x)

        return self.fc2(x)

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal((1, 32, 32, 3))
    output_tensor = model(input_tensor)
    print(output_tensor.shape)