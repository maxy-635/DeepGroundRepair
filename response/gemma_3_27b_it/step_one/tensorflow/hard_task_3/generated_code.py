import tensorflow as tf

class DL_Model(tf.keras.Model):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Define the layers
        self.conv1_group1 = tf.keras.layers.Conv2D(filters=32, kernel_size=1, strides=1, padding='same', activation='relu')
        self.conv2_group1 = tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=1, padding='same', activation='relu')
        self.dropout1 = tf.keras.layers.Dropout(rate=0.2)

        self.conv1_group2 = tf.keras.layers.Conv2D(filters=32, kernel_size=1, strides=1, padding='same', activation='relu')
        self.conv2_group2 = tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=1, padding='same', activation='relu')
        self.dropout2 = tf.keras.layers.Dropout(rate=0.2)

        self.conv1_group3 = tf.keras.layers.Conv2D(filters=32, kernel_size=1, strides=1, padding='same', activation='relu')
        self.conv2_group3 = tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=1, padding='same', activation='relu')
        self.dropout3 = tf.keras.layers.Dropout(rate=0.2)

        self.concat = tf.keras.layers.Concatenate(axis=-1)

        self.conv1_branch = tf.keras.layers.Conv2D(filters=64, kernel_size=1, strides=1, padding='same', activation='relu')

        self.add = tf.keras.layers.Add()

        self.dense = tf.keras.layers.Dense(units=10, activation='softmax')

    def call(self, inputs):
        # Main Pathway
        group1 = self.conv1_group1(inputs)
        group1 = self.conv2_group1(group1)
        group1 = self.dropout1(group1)

        group2 = self.conv1_group2(inputs)
        group2 = self.conv2_group2(group2)
        group2 = self.dropout2(group2)

        group3 = self.conv1_group3(inputs)
        group3 = self.conv2_group3(group3)
        group3 = self.dropout3(group3)

        main_pathway = self.concat([group1, group2, group3])

        # Branch Pathway
        branch_pathway = self.conv1_branch(inputs)

        # Combine Pathways
        output = self.add([main_pathway, branch_pathway])

        # Classification
        output = self.dense(output)

        return output


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal(shape=(1, 32, 32, 3))  # CIFAR-10 input shape: (batch_size, height, width, channels)
    output_tensor = model(input_tensor)
    print(output_tensor.shape)