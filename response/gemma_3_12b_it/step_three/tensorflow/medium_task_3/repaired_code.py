import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPool2D, Add, Dense, Input
from tensorflow.keras.models import Model

class DL_Model(Model):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Block 1: Conv2D -> MaxPool2D
        self.conv1_block1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', name='conv1_block1')
        self.pool1_block1 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid', name='pool1_block1')

        # Block 2: Conv2D -> MaxPool2D
        self.conv1_block2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', name='conv1_block2')
        self.pool1_block2 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid', name='pool1_block2')

        # Fully connected layer
        self.dense1 = Dense(units=10, activation='softmax', name='dense1')

    def call(self, inputs):
        # Path 1: Block 1
        x1 = self.conv1_block1(inputs)
        x1 = self.pool1_block1(x1)

        # Path 2: Block 2
        x2 = self.conv1_block2(inputs)
        x2 = self.pool1_block2(x2)

        # Combine the paths with the input
        merged = Add()([inputs, x1, x2])

        # Flatten the output
        flattened = tf.keras.layers.Flatten()(merged)

        # Fully connected layer
        output = self.dense1(flattened)
        return output


if __name__ == "__main__":
    model = DL_Model()
    # Dummy input tensor for MNIST (28x28 grayscale images)
    input_tensor = Input(shape=(28, 28, 1))
    
    # Build the model using the input tensor
    #output_tensor = model(input_tensor)
    
    # To correctly build the model with the input tensor, you need to define the call method in the DL_Model class.
    # The call method should take the input tensor as an argument and return the output tensor.
    # The code above already defines the call method.
    
    # Now, you can compile and train the model.
    # For example:
    # model.compile(optimizer='adam',
    #               loss='sparse_categorical_crossentropy',
    #               metrics=['accuracy'])
    #
    # You would then need to load the MNIST dataset and train the model.
    
    # The following lines are for validating the model structure
    model.build(input_shape=(None, 28, 28, 1))
    output_tensor = model(input_tensor)
    print("Model built and input tensor processed successfully.")