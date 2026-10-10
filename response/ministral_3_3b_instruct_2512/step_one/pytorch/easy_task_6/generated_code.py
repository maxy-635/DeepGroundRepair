import torch
import torch.nn as nn

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()
        # Main path: two convolutional layers
        self.main_path = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1),  # Increase feature map width
            nn.Conv2d(in_channels=32, out_channels=1, kernel_size=3, stride=1, padding=1)   # Restore channels
        )

        # Branch path: direct connection
        self.branch_path = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=1, stride=1, padding=0)

        # Flatten and fully connected layer
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(in_features=28*28, out_features=10)

    def forward(self, x):
        # Main path
        main_output = self.main_path(x)

        # Branch path
        branch_output = self.branch_path(x)

        # Add the two paths
        combined = torch.add(main_output, branch_output)

        # Flatten and pass through fully connected layer
        output = self.flatten(combined)
        output = self.fc(output)

        return output

if __name__ == "__main__":
    # Create model instance
    model = DL_Model()

    # Create dummy input tensor with shape (1, 28, 28)
    input_tensor = torch.randn(1, 1, 28, 28)

    # Forward pass
    output_tensor = model(input_tensor)
    print("Input shape:", input_tensor.shape)
    print("Output shape:", output_tensor.shape)