import torch
import torch.nn as nn

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Initial convolutional layer with 32 kernels
        self.initial_conv = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)

        # Block 1
        self.block1_conv1 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=1, stride=1, padding=0)
        self.block1_depthwise_conv = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.block1_conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=1, stride=1, padding=0)

        # Block 2
        self.permute = nn.Permute((0, 3, 1, 2))  # Swap the third and fourth dimensions
        self.reshape = nn.Flatten(start_dim=1)  # Flatten the input to (batch_size, channels, height*width)

        # Fully connected layer for classification
        self.fc = nn.Linear(in_features=32 * 28 * 28, out_features=10)

    def forward(self, x):
        # Initial convolutional layer
        x = self.initial_conv(x)

        # Block 1
        x1, x2 = torch.split(x, split_size_or_sections=16, dim=2)
        x1 = self.block1_conv1(x1)
        x1 = self.block1_depthwise_conv(x1)
        x1 = self.block1_conv2(x1)
        x2 = self.block1_conv1(x2)  # Pass through without modification
        x = torch.cat((x1, x2), dim=1)

        # Block 2
        # Reshape the input to (batch_size, channels, height, width)
        batch_size, channels, height, width = x.shape
        # Reshape into four groups
        reshaped_x = x.view(batch_size, channels, height, width)
        # Swap the third and fourth dimensions
        reshaped_x = self.permute(reshaped_x)
        # Reshape back to original shape for channel shuffling
        reshaped_x = reshaped_x.view(batch_size, channels, height, width)

        # Flatten and pass through fully connected layer
        flattened_x = self.reshape(reshaped_x)
        output = self.fc(flattened_x)

        return output

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 1, 28, 28)  # Dummy input tensor with shape (batch_size, channels, height, width)
    output_tensor = model(input_tensor)
    print("Input shape:", input_tensor.shape)
    print("Output shape:", output_tensor.shape)