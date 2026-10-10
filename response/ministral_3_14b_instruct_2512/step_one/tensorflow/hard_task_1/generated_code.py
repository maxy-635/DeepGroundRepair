import tensorflow as tf
from tensorflow.keras.layers import Conv2D, GlobalAveragePooling2D, GlobalMaxPooling2D, Dense, Add, Activation, Multiply, Concatenate, Lambda
from tensorflow.keras.models import Model

class DL_Model(Model):
    def __init__(self, input_shape=(32, 32, 3), num_classes=10):
        super(DL_Model, self).__init__()
        self.input_shapes = input_shape
        self.num_classes = num_classes

        # Initial convolutional layer
        self.initial_conv = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')

        # Block 1: Channel Attention
        self.global_avg_pool = GlobalAveragePooling2D(keepdims=True)
        self.global_max_pool = GlobalMaxPooling2D(keepdims=True)
        self.fc1 = Dense(units=64, activation='relu')
        self.fc2 = Dense(units=64, activation='relu')
        self.fc3 = Dense(units=64, activation='relu')
        self.fc4 = Dense(units=64, activation='relu')
        self.sigmoid = Activation('sigmoid')

        # Block 2: Spatial Attention
        self.avg_pool = GlobalAveragePooling2D()
        self.max_pool = GlobalMaxPooling2D()
        self.conv1x1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='sigmoid')

        # Additional branch
        self.conv1x1_branch = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same')

        # Final classification layer
        self.fc_final = Dense(units=num_classes, activation='softmax')

    def call(self, inputs):
        # Initial convolutional layer
        x = self.initial_conv(inputs)

        # Block 1: Channel Attention
        # Path1: Global average pooling + FC layers
        avg_path = self.global_avg_pool(x)
        avg_path = self.fc1(avg_path)
        avg_path = self.fc2(avg_path)

        # Path2: Global max pooling + FC layers
        max_path = self.global_max_pool(x)
        max_path = self.fc3(max_path)
        max_path = self.fc4(max_path)

        # Add paths and apply sigmoid
        attention = Add()([avg_path, max_path])
        attention = self.sigmoid(attention)

        # Reshape attention to match input channels
        attention = Lambda(lambda x: tf.reshape(x, (-1, 1, 1, 64)))(attention)

        # Apply channel attention
        x = Multiply()([x, attention])

        # Block 2: Spatial Attention
        avg_pool = GlobalAveragePooling2D()(x)
        max_pool = GlobalMaxPooling2D()(x)

        # Concatenate and apply 1x1 conv
        concat = Concatenate(axis=-1)([avg_pool, max_pool])
        concat = Lambda(lambda x: tf.expand_dims(x, axis=1))(concat)
        concat = Lambda(lambda x: tf.expand_dims(x, axis=1))(concat)
        concat = self.conv1x1(concat)

        # Reshape to match input shape
        concat = Lambda(lambda x: tf.reshape(x, (-1, 1, 1, 64)))(concat)

        # Apply spatial attention
        x = Multiply()([x, concat])

        # Additional branch
        branch = self.conv1x1_branch(x)
        x = Add()([x, branch])

        # Final activation
        x = Activation('relu')(x)

        # Global average pooling for classification
        x = GlobalAveragePooling2D()(x)

        # Final classification layer
        output = self.fc_final(x)

        return output

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal((1, 32, 32, 3))
    output_tensor = model(input_tensor)
    print("Input shape:", input_tensor.shape)
    print("Output shape:", output_tensor.shape)