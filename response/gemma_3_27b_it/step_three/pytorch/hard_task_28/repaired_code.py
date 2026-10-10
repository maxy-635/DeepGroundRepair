import torch
import torch.nn as nn

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Main Path
        self.main_path = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=7, stride=1, padding=3),
            nn.LayerNorm(normalized_shape=[3, 32, 32]),
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1, stride=1, padding=0)
        )

        # Branch Path (direct connection to input)

        # Combination and Classification
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_features=3 * 32 * 32, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=10)  # 10 classes for CIFAR-10

    def forward(self, x):
        # the initial shape of x is: (1, 3, 32, 32)
        main_output = self.main_path(x)
        branch_output = x
        combined_output = main_output + branch_output
        flattened_output = self.flatten(combined_output)
        fc1_output = torch.relu(self.fc1(flattened_output))
        output = self.fc2(fc1_output)
        return output


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 3, 32, 32)  # Batch size of 1, 3 channels, 32x32 image
    output_tensor = model(input_tensor)
    print(output_tensor.shape)  # Expected output shape: torch.Size([1, 10])