import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Add, GlobalAveragePooling2D, Dense, Activation, Multiply, Lambda, Concatenate, Reshape
from tensorflow.keras.models import Model

class DL_Model(Model):
    def __init__(self, num_classes=10):
        super(DL_Model, self).__init__()
        # Branch 1: 3x3 convolution
        self.conv3x3_1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')
        self.conv3x3_2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')

        # Branch 2: 5x5 convolution
        self.conv5x5_1 = Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')
        self.conv5x5_2 = Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')

        # Attention mechanism
        self.pool = GlobalAveragePooling2D()
        self.dense1 = Dense(units=128, activation='relu')
        self.dense2 = Dense(units=2, activation='softmax')

        # Final layers
        self.fc = Dense(units=num_classes, activation='softmax')

    def call(self, inputs):
        # Branch 1 processing
        x3x3 = self.conv3x3_1(inputs)
        x3x3 = self.conv3x3_2(x3x3)

        # Branch 2 processing
        x5x5 = self.conv5x5_1(inputs)
        x5x5 = self.conv5x5_2(x5x5)

        # Combine branches with addition
        combined = Add()([x3x3, x5x5])

        # Attention mechanism
        attention = self.pool(combined)
        attention = self.dense1(attention)
        attention = self.dense2(attention)

        # Reshape attention weights for multiplication
        attention_3x3 = Lambda(lambda x: x[:, 0])(attention)
        attention_5x5 = Lambda(lambda x: x[:, 1])(attention)

        # Apply attention weights
        weighted_3x3 = Multiply()([x3x3, attention_3x3])
        weighted_5x5 = Multiply()([x5x5, attention_5x5])

        # Combine weighted outputs
        weighted_combined = Add()([weighted_3x3, weighted_5x5])

        # Global average pooling and final classification
        pooled = GlobalAveragePooling2D()(weighted_combined)
        output = self.fc(pooled)

        return output

if __name__ == "__main__":
    # Create model instance
    model = DL_Model()

    # Create dummy input tensor with shape matching CIFAR-10 (32x32x3)
    input_tensor = tf.random.normal(shape=(1, 32, 32, 3))

    # Get model output
    output_tensor = model(input_tensor)

    print("Input shape:", input_tensor.shape)
    print("Output shape:", output_tensor.shape)