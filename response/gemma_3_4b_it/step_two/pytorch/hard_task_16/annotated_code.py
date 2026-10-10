import torch
import torch.nn as nn

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Block 1
        self.block1_conv1 = nn.Conv1d(3, 32, kernel_size=1, stride=1, padding=0)
        self.block1_conv2 = nn.Conv1d(32, 32, kernel_size=3, stride=1, padding=1)
        self.block1_conv3 = nn.Conv1d(32, 32, kernel_size=1, stride=1, padding=0)
        self.block1_concat = nn.Concat('mask')

        # Transition Convolution
        self.transition_conv = nn.Conv1d(32, 16, kernel_size=1, stride=1, padding=0)

        # Block 2
        self.block2_pool = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
        self.block2_fc1 = nn.Linear(16, 128)
        self.block2_fc2 = nn.Linear(128, 10)
        self.block2_weight = nn.Linear(16, 16)

        # Branch
        self.branch_fc1 = nn.Linear(16, 128)
        self.branch_fc2 = nn.Linear(128, 10)

    def forward(self, x):
        # Block 1
        x = self.block1_concat

        # Transition Convolution
        x = self.transition_conv(x)

        # Block 2
        x = self.block2_pool(x)
        x = torch.sigmoid(self.block2_fc1(x))
        weight = self.block2_weight(x)
        x = x * weight
        x = self.block2_fc2(x)

        # Branch
        branch = x
        branch = self.branch_fc1(branch)
        branch = self.branch_fc2(branch)

        # Add main path and branch
        x = x + branch

        return x

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 3, 32)  # Example input shape: (batch_size, channels, spatial_size)
    output_tensor = model(input_tensor)
    print(output_tensor.shape)