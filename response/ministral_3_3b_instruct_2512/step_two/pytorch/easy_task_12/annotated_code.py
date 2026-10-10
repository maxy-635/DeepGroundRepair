import torch
import torch.nn as nn

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Main path
        self.main_path = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Branch path
        self.branch_path = nn.Sequential(
            nn.Conv1d(in_channels='mask', out_channels='mask', kernel_size='mask', stride='mask', padding='mask'),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.AdaptiveAvgPool2d(output_size=(1, 1))
        )

        # 1x1 convolution to match dimensions
        self.match_conv = nn.Conv2d(64, 64, kernel_size=1, stride=1)

        # Flatten and fully connected layer
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(64, 10)

    def forward(self, x):
        # the initial shape of x is: (1, 1, 28, 28)
        # Main path
        main_output = self.main_path(x)  # output tensor shape: (1, 64, 14, 14)

        # Branch path
        branch_output = self.branch_path(x)
        branch_output = branch_output.view(branch_output.size(0), 64, 1, 1)

        # Match dimensions and sum
        matched_output = self.match_conv(branch_output)
        combined_output = main_output + matched_output

        # Flatten and fully connected
        output = self.flatten(combined_output)
        output = self.fc(output)

        return output

if __name__ == "__main__":
    # Create a dummy input tensor with shape (batch_size, channels, height, width)
    input_tensor = torch.randn(1, 1, 28, 28)
    model = DL_Model()
    output_tensor = model(input_tensor)
    print("Input shape:", input_tensor.shape)
    print("Output shape:", output_tensor.shape)