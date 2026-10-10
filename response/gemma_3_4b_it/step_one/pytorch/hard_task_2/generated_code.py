import torch
import torch.nn as nn

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()

        # First group of convolutions
        self.conv1 = nn.Conv1d(3, 16, kernel_size=1, stride=1, padding=0)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv1d(32, 16, kernel_size=1, stride=1, padding=0)

        # Main path
        self.main_path = nn.Sequential(
            nn.Conv1d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(32, 16, kernel_size=1, stride=1, padding=0)
        )

        # Fusion
        self.fusion = nn.Conv1d(16, 32, kernel_size=3, stride=1, padding=1)
        self.relu_fusion = nn.ReLU(inplace=True)

        # Flatten and linear layer
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(32, 10)

    def forward(self, x):
        # First group
        x1 = self.conv1(x)
        x1 = self.relu1(x1)
        x1 = self.conv2(x1)
        x1 = self.relu2(x1)
        x1 = self.conv3(x1)

        # Main path
        main_path_output = self.main_path(x)

        # Fusion
        x_combined = torch.add(x1, main_path_output)
        x_combined = self.fusion(x_combined)
        x_combined = self.relu_fusion(x_combined)

        # Flatten and linear
        x_flattened = self.flatten(x_combined)
        x_output = self.linear(x_flattened)

        return x_output


if __name__ == "__main__":
    model = DL_Model()
    # Dummy input tensor for validation
    input_tensor = torch.randn(1, 3, 32)  # Batch size 1, 3 channels, 32 spatial dimensions
    output_tensor = model(input_tensor)
    print(output_tensor.shape)