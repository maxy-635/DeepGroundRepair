import torch
import torch.nn as nn
import torch.optim as optim

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()

        # First block: Dual-path structure
        self.main_path = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 3, kernel_size=3, stride=1, padding=1)
        )
        self.branch_path = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )

        # Second block: Channel splitting
        self.channel_split = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=5, stride=1, padding=2)
        )

        # Fully connected layers
        self.fc1 = nn.Linear(32 * 3 * 3, 10)  # Adjust the input size based on the convolutional output
        self.fc2 = nn.Linear(10, 10)

    def forward(self, x):
        # Main path
        main_output = self.main_path(x)

        # Branch path
        branch_output = self.branch_path(x)

        # Combine paths
        combined_output = torch.add(main_output, branch_output)

        # Channel splitting
        channel_split_output = self.channel_split(combined_output)

        # Concatenate the outputs
        concatenated_output = torch.cat([main_output, channel_split_output], dim=1)

        # Fully connected layers
        output = self.fc1(concatenated_output)
        output = self.fc2(output)

        return output

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 3, 32, 32)  # Batch size 1, 3 channels, 32x32 image
    output_tensor = model(input_tensor)
    print(output_tensor.shape)