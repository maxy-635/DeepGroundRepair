import paddle
import paddle.nn as nn
import paddle.io
import paddle.vision.datasets as datasets

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()

        # First convolutional layer (1x1)
        self.conv1 = nn.Conv2D(in_channels=3, out_channels=64, kernel_size=1, stride=1, padding=0)

        # Second convolutional layer (3x3)
        self.conv2 = nn.Conv2D(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)

        # Third convolutional layer (5x5)
        self.conv3 = nn.Conv2D(in_channels=64, out_channels=64, kernel_size=5, stride=1, padding=2)

        # Concatenation operation
        self.concat = nn.Concat(dim=1)

        # Flatten layer
        self.flatten = nn.Flatten()

        # Fully connected layer 1
        self.fc1 = nn.Linear(in_features=64 * 64 * 64, out_features=128)

        # Fully connected layer 2
        self.fc2 = nn.Linear(in_features=128, out_features=10)

        # Cross entropy loss
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        # Convolutional layers
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        # Concatenate the outputs
        x = self.concat(x)

        # Flatten the feature maps
        x = self.flatten(x)

        # Fully connected layers
        x = self.fc1(x)
        x = self.fc2(x)

        return x


if __name__ == "__main__":
    # Instantiate the model
    model = DL_Model()

    # Create a dummy input tensor
    input_tensor = paddle.randn(batch_size=1, channels=3, height=32, width=32)

    # Pass the input tensor through the model
    output_tensor = model(input_tensor)

    # Print the output shape
    print("Output shape:", output_tensor.shape)