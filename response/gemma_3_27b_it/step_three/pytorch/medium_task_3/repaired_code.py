import torch
import torch.nn as nn

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()

        # First convolutional block
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Second convolutional block
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        # Flatten layer
        self.flatten = nn.Flatten()

        # Fully connected layer
        self.fc = nn.Linear(in_features=32 * 7 * 7, out_features=10)
    
    def forward(self, x):
        # the initial shape of x is: (1, 1, 28, 28)
        # Pass through the first convolutional block
        x1 = self.pool1(torch.relu(self.conv1(x)))  # output tensor shape: (1, 32, 14, 14)

        # Pass through the second convolutional block
        x2 = self.pool2(torch.relu(self.conv2(x1)))  # output tensor shape of this layer should be (1, 32, 7, 7)
        
        # Add the original input to the output of the second block
        x_combined = torch.add(x, x2)

        # Flatten the combined output
        x_flattened = self.flatten(x_combined)

        # Pass through the fully connected layer
        output = self.fc(x_flattened)
        return output


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 1, 28, 28)  # Batch size of 1, 1 channel, 28x28 image
    output_tensor = model(input_tensor)
    print(output_tensor.shape)