import torch
import torch.nn as nn

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=2)
        self.relu1 = nn.ReLU(inplace=True)
        self.relu2 = nn.ReLU(inplace=True)
        self.relu3 = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(kernel_size=8, stride=8, padding=0)
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(128 * 8 * 8, 10)

    def forward(self, x):
        # the initial shape of x is: (4, 3, 32, 32)
        # Split the input into three groups
        x1 = x[:, :1, :, :]  # output tensor shape: (4, 1, 32, 32)
        x2 = x[:, 1:2, :, :]  # output tensor shape: (4, 1, 32, 32)
        x3 = x[:, 2:, :, :]  # output tensor shape: (4, 1, 32, 32)

        # Feature extraction via depthwise separable convolutional layers
        x1 = self.conv1(x1)
        x1 = self.relu1(x1)
        x2 = self.conv2(x2)
        x2 = self.relu2(x2)
        x3 = self.conv3(x3)
        x3 = self.relu3(x3)

        # Concatenate the outputs
        x = torch.cat((x1, x2, x3), dim=1)

        # Average pooling
        x = self.avgpool(x)

        # Flatten
        x = self.flatten(x)

        # Linear layer for classification
        x = self.linear(x)

        return x


if __name__ == "__main__":
    # Define the input shape for CIFAR-10
    batch_size = 4
    image_size = 32
    input_channel = 3
    num_classes = 10

    # Instantiate the DL_Model
    model = DL_Model()

    # Create a dummy input tensor
    input_tensor = torch.randn(batch_size, input_channel, image_size, image_size)

    # Pass the input tensor through the model
    output_tensor = model(input_tensor)

    # Print the output shape
    print("Output shape:", output_tensor.shape)