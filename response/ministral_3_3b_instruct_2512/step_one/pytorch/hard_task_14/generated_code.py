import torch
import torch.nn as nn

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Main path: Global Average Pooling followed by two fully connected layers
        self.main_path = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            nn.Flatten(),
            nn.Linear(in_features=32, out_features=32),  # Assuming input channels are 32
            nn.Linear(in_features=32, out_features=32)
        )

        # Branch path: 3x3 convolution to match input channels
        self.branch_path = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )

        # Reshape weights from main path to match input shape
        self.weights = nn.Parameter(torch.randn(32, 32, 1, 1), requires_grad=True)

        # Final fully connected layers
        self.final_layers = nn.Sequential(
            nn.Linear(in_features=32, out_features=1024),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=1024, out_features=512),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=512, out_features=10)
        )

    def forward(self, x):
        # Main path processing
        main_output = self.main_path(x)
        # Reshape weights to match input dimensions
        weights_reshaped = self.weights.view(32, 32, 1, 1)
        # Element-wise multiplication with original feature map
        elementwise_product = torch.add(x, weights_reshaped * main_output)

        # Branch path processing
        branch_output = self.branch_path(x)

        # Add outputs from main and branch paths
        combined_output = torch.add(elementwise_product, branch_output)

        # Flatten combined output for final layers
        combined_flattened = combined_output.flatten(start_dim=1)

        # Final classification layers
        output = self.final_layers(combined_flattened)

        return output

if __name__ == "__main__":
    # Create a dummy input tensor with shape (batch_size, channels, height, width)
    # Assuming CIFAR-10 images are 32x32 with 3 channels
    input_tensor = torch.randn(1, 32, 32, 32)  # Example with batch size 1, 32 channels, 32x32 image
    output_tensor = DL_Model()(input_tensor)
    print("Input shape:", input_tensor.shape)
    print("Output shape:", output_tensor.shape)