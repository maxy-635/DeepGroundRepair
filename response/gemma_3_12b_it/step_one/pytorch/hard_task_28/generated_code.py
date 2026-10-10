import torch
import torch.nn as nn

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()
        # Main path
        self.main_path = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=7, stride=1, padding=3, bias=False),
            nn.LayerNorm(normalized_shape=[3, 1, 1]),
            nn.Sequential(
                nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1, stride=1, padding=0, bias=False),
                nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1, stride=1, padding=0, bias=False)
            )
        )
        # Branch path
        self.branch_path = nn.Identity()  # Simply returns the input

        # Combine paths
        self.combine = torch.add

        # Classification layers
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_features=3 * 32 * 32, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=10)

    def forward(self, x):
        # Main path
        main_output = self.main_path(x)
        # Branch path
        branch_output = self.branch_path(x)
        # Combine paths
        combined_output = self.combine(main_output, branch_output)
        # Flatten and classification
        flattened_output = self.flatten(combined_output)
        fc1_output = self.fc1(flattened_output)
        output = self.fc2(fc1_output)
        return output


if __name__ == "__main__":
    model = DL_Model()
    # Dummy input tensor (CIFAR-10: 32x32 images with 3 channels)
    input_tensor = torch.randn(1, 3, 32, 32)  # Batch size of 1
    output_tensor = model(input_tensor)
    print(output_tensor.shape)