import paddle
import paddle.nn as nn

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()
        # Define the convolutional layer with 32 output channels, 3x3 kernel size, stride 1, and padding 1
        self.conv1 = nn.Conv2D(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        # Define the max pooling layer with 2x2 kernel size and stride 2
        self.pool1 = nn.MaxPool2D(kernel_size=2, stride=2)
        # Define the second convolutional layer with 64 output channels, 3x3 kernel size, stride 1, and padding 1
        self.conv2 = nn.Conv2D(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        # Define the flatten layer to flatten the feature maps into a one-dimensional vector
        self.flatten = nn.Flatten()
        # Define the first fully connected layer with 128 output features
        self.fc1 = nn.Linear(in_features=64 * 7 * 7, out_features=128)
        # Define the second fully connected layer with 10 output features (for MNIST classes)
        self.fc2 = nn.Linear(in_features=128, out_features=10)

    def forward(self, x):
        # Apply the first convolutional layer and ReLU activation
        x = paddle.nn.functional.relu(self.conv1(x))
        # Apply the first max pooling layer
        x = self.pool1(x)
        # Apply the second convolutional layer and ReLU activation
        x = paddle.nn.functional.relu(self.conv2(x))
        # Apply the max pooling layer again
        x = self.pool1(x)
        # Flatten the feature maps
        x = self.flatten(x)
        # Apply the first fully connected layer with ReLU activation
        x = paddle.nn.functional.relu(self.fc1(x))
        # Apply dropout with a probability of 0.5
        x = nn.Dropout(p=0.5)(x)
        # Apply the second fully connected layer
        x = self.fc2(x)
        return x

if __name__ == "__main__":
    model = DL_Model()
    # Create a dummy input tensor with shape (batch_size=1, channels=1, height=28, width=28)
    input_tensor = paddle.randn((1, 1, 28, 28))
    output_tensor = model(input_tensor)
    print("Input tensor shape:", input_tensor.shape)
    print("Output tensor shape:", output_tensor.shape)