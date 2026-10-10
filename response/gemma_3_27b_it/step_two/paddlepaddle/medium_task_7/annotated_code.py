import paddle
import paddle.nn as nn

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Define the convolutional layers
        self.conv1 = nn.Conv2D(in_channels=3, out_channels='mask', kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2D(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2D(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv_direct = nn.Conv2D(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)

        # Define the ReLU activation function
        self.relu = nn.ReLU()

        # Define the fully connected layers
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_features=64 * 32 * 32, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=10)

    def forward(self, x):
        # the initial shape of x is: (1, 3, 32, 32)
        # Pass the input through the first two convolutional layers
        x1 = self.conv1(x)  # output tensor shape of this layer should be (1, 64, 32, 32)
        x1 = self.relu(x1)
        x2 = self.conv2(x1)  # output tensor shape: (1, 64, 32, 32)
        x2 = self.relu(x2)  # output tensor shape: (1, 64, 32, 32)

        # Pass the input directly through a convolutional layer
        x_direct = self.conv_direct(x)  # output tensor shape: (1, 64, 32, 32)
        x_direct = self.relu(x_direct)  # output tensor shape: (1, 64, 32, 32)

        # Pass the output of the second convolutional layer through the third convolutional layer
        x3 = self.conv3(x2)  # output tensor shape: (1, 64, 32, 32)
        x3 = self.relu(x3)  # output tensor shape: (1, 64, 32, 32)

        # Add the outputs of the first two convolutional layers and the third convolutional layer
        x_added = x1 + x3

        # Add the direct convolutional output to the added output
        x_combined = x_added + x_direct

        # Flatten the combined output
        x_flattened = self.flatten(x_combined)

        # Pass the flattened output through the fully connected layers
        x_fc1 = self.fc1(x_flattened)
        x_fc1 = self.relu(x_fc1)
        x_fc2 = self.fc2(x_fc1)

        return x_fc2


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn(shape=(1, 3, 32, 32))  # Batch size of 1, 3 channels, 32x32 image
    output_tensor = model(input_tensor)
    print(output_tensor.shape)