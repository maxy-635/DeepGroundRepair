import tensorflow as tf

class DL_Model(tf.keras.Model):
    def __init__(self):
        super(DL_Model, self).__init__()
        # Main pathway with three parallel branches
        self.main_branch1 = tf.keras.Sequential([
            tf.keras.layers.Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu'),
            tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')
        ])

        self.main_branch2 = tf.keras.Sequential([
            tf.keras.layers.Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu'),
            tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')
        ])

        self.main_branch3 = tf.keras.Sequential([
            tf.keras.layers.Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu'),
            tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')
        ])

        self.dropout1 = tf.keras.layers.Dropout(rate=0.5)

        # Branch pathway
        self.branch_branch = tf.keras.Sequential([
            tf.keras.layers.Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')
        ])

        # Concatenate the main pathway outputs
        self.concatenate = tf.keras.layers.concatenate(axis='mask')

        # Add the branch pathway output
        self.add = tf.keras.layers.Add()

        # Final dense layer
        self.dense = tf.keras.layers.Dense(units=10, activation='softmax')

    def call(self, inputs):
        # Process each branch of the main pathway
        branch1 = self.main_branch1(inputs)
        branch2 = self.main_branch2(inputs)
        branch3 = self.main_branch3(inputs)

        # Concatenate the three branches
        concatenated = self.concatenate([branch1, branch2, branch3])

        # Apply dropout
        output = self.dropout1(concatenated)

        # Process the branch pathway
        branch_output = self.branch_branch(inputs)

        # Combine using addition
        combined = self.add([output, branch_output])

        # Final classification
        return self.dense(combined)

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal(shape=(1, 32, 32, 3))  # Example input shape for CIFAR-10
    output_tensor = model(input_tensor)
    print("Input shape:", input_tensor.shape)
    print("Output shape:", output_tensor.shape)