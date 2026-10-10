import torch
import torch.nn as nn

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Split input into three groups
        self.group1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=1, stride=1, padding=0)
        )

        self.group2 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=1, stride=1, padding=0)
        )

        self.group3 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=1, stride=1, padding=0)
        )

        # Combine outputs from three groups
        self.main_path = nn.Sequential(
            self.group1,
            self.group2,
            self.group3
        )

        # Fuse with original input
        self.fuse_layer = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=1, stride=1, padding=0)

        # Classification layers
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(in_features=32 * 32 * 32, out_features=10)  # Assuming CIFAR-10 input size of 32x32

    def forward(self, x):
        # Split input into three groups
        group1_output = self.group1(x)
        group2_output = self.group2(x)
        group3_output = self.group3(x)

        # Combine outputs from three groups
        combined_output = torch.add(group1_output, group2_output)
        combined_output = torch.add(combined_output, group3_output)

        # Fuse with original input
        fused_output = torch.add(combined_output, self.fuse_layer(x))

        # Flatten and classify
        flattened_output = self.flatten(fused_output)
        output = self.linear(flattened_output)

        return output


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 3, 32, 32)  # Batch size of 1, 3 channels, 32x32 image
    output_tensor = model(input_tensor)
    print(output_tensor.shape)