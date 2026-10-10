import torch
import torch.nn as nn

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()
        # Define the convolutional layers with various kernel sizes
        self.conv1_1x1 = nn.Conv2d(3, 64, kernel_size=1, stride=1, padding=0)
        self.conv1_3x3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv1_5x5 = nn.Conv2d(64, 64, kernel_size=5, kernel_size=5, stride=1, padding=2)

        # Define the pooling layer
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Define the concatenation layers
        self.concat1 = nn.Sequential(
            nn.Conv2d(192, 128, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128)
        )

        # Define the flattening layer
        self.flatten = nn.Flatten()

        # Define the fully connected layers
        self.fc1 = nn.Linear(128 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        # Apply 1x1 convolution
        x1 = self.conv1_1x1(x)

        # Apply 3x3 convolution
        x2 = self.conv1_3x3(x1)

        # Apply 5x5 convolution
        x3 = self.conv1_5x5(x2)

        # Concatenate the outputs of the convolutions
        x_concat = torch.cat((x1, x2, x3), dim=1)

        # Apply pooling
        x_pool = self.pool1(x_concat)

        # Apply concatenation layers
        x_concat2 = self.concat1(x_pool)

        # Flatten the output
        x_flatten = self.flatten(x_concat2)

        # Apply fully connected layers
        x_fc1 = self.fc1(x_flatten)
        x_fc2 = self.fc2(x_fc1)

        return x_fc2

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 3, 32, 32)  # Example input tensor with batch size 1, 3 channels, 32x32 image
    output_tensor = model(input_tensor)
    print(output_tensor.shape)  # Expected output shape: torch.Size([1, 10])