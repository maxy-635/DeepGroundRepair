import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPool2D, Add, GlobalAveragePooling2D, Dense, Multiply
from tensorflow.keras.activations import softmax
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy

class DL_Model(tf.keras.Model):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Convolutional branch 1 with 3x3 kernel
        self.conv2d_3x3_1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', name='conv2d_3x3_1')
        self.maxpool2d_3x3_1 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same', name='maxpool2d_3x3_1')

        # Convolutional branch 2 with 5x5 kernel
        self.conv2d_5x5_1 = Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu', name='conv2d_5x5_1')
        self.maxpool2d_5x5_1 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same', name='maxpool2d_5x5_1')

        # Feature combination
        self.addition = Add(name='addition')

        # Global average pooling
        self.global_avg_pool = GlobalAveragePooling2D(name='global_avg_pool')

        # Attention mechanism
        self.dense_attention_1 = Dense(units=32, activation='relu', name='dense_attention_1')
        self.softmax_attention = softmax(name='attention_weights')
        self.dense_attention_2 = Dense(units=10, activation='relu', name='dense_attention_2')

        # Weighted output
        self.multiply_1 = Multiply(name='multiply_1')
        self.multiply_2 = Multiply(name='multiply_2')

        # Final classification
        self.dense_classification = Dense(units=10, activation='softmax', name='dense_classification')

    def call(self, inputs):
        # Branch 1
        x1 = self.conv2d_3x3_1(inputs)
        x1 = self.maxpool2d_3x3_1(x1)

        # Branch 2
        x2 = self.conv2d_5x5_1(inputs)
        x2 = self.maxpool2d_5x5_1(x2)

        # Combine features
        combined_features = self.addition([x1, x2])

        # Global average pooling
        pooled_features = self.global_avg_pool(combined_features)

        # Attention mechanism
        attention_weights = self.dense_attention_1(pooled_features)
        attention_weights = self.softmax_attention(attention_weights)
        attention_weights = self.dense_attention_2(attention_weights)

        # Weighted output
        weighted_output_1 = self.multiply_1([x1, attention_weights])
        weighted_output_2 = self.multiply_2([x2, attention_weights])

        # Final classification
        output = self.dense_classification(pooled_features)

        return output


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal((1, 32, 32, 3))  # Example input shape (batch_size, height, width, channels)
    output_tensor = model(input_tensor)
    print("Output shape:", output_tensor.shape)