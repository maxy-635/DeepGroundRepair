import torch
import torch.nn as nn

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()
        # Main path: Global Average Pooling followed by two fully connected layers
        self.main_path = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            nn.Flatten(),
            nn.Linear(in_features=32, out_features=32),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=32, out_features=32),
            nn.ReLU(inplace=True)
        )

        # Branch path: Directly connected to the input layer
        self.branch_path = nn.Sequential(
            nn.Flatten()
        )

        # Combine the outputs from both paths
        self.combine = nn.Sequential(
            nn.Linear(in_features=32, out_features=32),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=32, out_features=10)  # Assuming 10 classes in CIFAR-10
        )

    def forward(self, x):
        # the initial shape of x is: (1, 3, 32, 32)
        # Main path processing
        main_output = self.main_path(x)

        # Branch path processing
        branch_output = self.branch_path(x)

        # Element-wise multiplication of main_output and branch_output
        combined_output = main_output + branch_output

        # Final processing
        output = self.combine(combined_output)
        return output

if __name__ == "__main__":
    # Create a dummy input tensor with shape (batch_size, channels, height, width)
    # Assuming CIFAR-10 images are 32x32 with 3 channels
    input_tensor = torch.randn(1, 3, 32, 32)
    model = DL_Model()
    output_tensor = model(input_tensor)
    print("Input shape:", input_tensor.shape)
    print("Output shape:", output_tensor.shape)