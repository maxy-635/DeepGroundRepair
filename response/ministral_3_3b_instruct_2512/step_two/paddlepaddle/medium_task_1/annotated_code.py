import paddle
import paddle.nn as nn

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()
        # Two convolutional layers
        self.conv1 = nn.Conv2D(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2D(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)

        # Max pooling layer
        self.max_pool = nn.MaxPool2D(kernel_size=2, stride=2, padding=0)

        # Layer normalization
        self.layer_norm = nn.LayerNorm(normalized_shape='mask')

        # Flatten layer
        self.flatten = nn.Flatten()

        # Two fully connected layers
        self.fc1 = nn.Linear(in_features=64 * 8 * 8, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=10)

        # Activation functions
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(axis=1)

    def forward(self, x):
        # the initial shape of x is: (1, 3, 32, 32)
        # First convolutional layer
        x = self.conv1(x)  # output tensor shape: (1, 32, 32, 32)
        x = self.relu(x)  # output tensor shape: (1, 32, 32, 32)

        # Second convolutional layer
        x = self.conv2(x)  # output tensor shape: (1, 64, 32, 32)
        x = self.relu(x)  # output tensor shape: (1, 64, 32, 32)

        # Max pooling
        x = self.max_pool(x)  # output tensor shape: (1, 64, 16, 16)

        # Add input features to the output of the convolutional layers
        x = x + x  # output tensor shape: (1, 64, 16, 16)

        # Layer normalization
        x = self.layer_norm(x)

        # Flatten
        x = self.flatten(x)

        # First fully connected layer
        x = self.fc1(x)
        x = self.relu(x)

        # Second fully connected layer
        x = self.fc2(x)

        return x

if __name__ == "__main__":
    model = DL_Model()
    # Assuming input tensor shape is (batch_size, channels, height, width)
    input_tensor = paddle.randn((1, 3, 32, 32))  # Example input tensor for CIFAR-10
    output_tensor = model(input_tensor)
    print("Input shape:", input_tensor.shape)
    print("Output shape:", output_tensor.shape)