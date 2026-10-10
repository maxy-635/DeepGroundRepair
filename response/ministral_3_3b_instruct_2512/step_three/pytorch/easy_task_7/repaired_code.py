import torch
import torch.nn as nn

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()
        self.main_path = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.Dropout(p=0.2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.Dropout(p=0.2),
            nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, stride=1, padding=1)
        )
        self.branch_path = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=1, stride=1, padding=0)
        )
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(in_features=784, out_features=10)

    def forward(self, x):
        # the initial shape of x is: (1, 1, 28, 28)
        main_path_output = self.main_path(x)  # output tensor shape: (1, 1, 28, 28)
        branch_path_output = self.branch_path(x)  # output tensor shape: (1, 1, 28, 28)
        combined_output = torch.add(main_path_output, branch_path_output)  # output tensor shape: (1, 1, 28, 28)
        flattened_output = self.flatten(combined_output)  # output tensor shape: (1, 784)
        output = self.fc(flattened_output)
        return output

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 1, 28, 28)  # Dummy input tensor with shape (batch_size, channels, height, width)
    output_tensor = model(input_tensor)
    print("Input shape:", input_tensor.shape)
    print("Output shape:", output_tensor.shape)