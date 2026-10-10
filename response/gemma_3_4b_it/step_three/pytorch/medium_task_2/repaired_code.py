import torch
import torch.nn as nn

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()
        self.main_path = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        )
        self.branch_path = nn.Sequential(
            nn.Conv2d(3, 16, 5, stride=1, padding=2)
        )
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(32 * 32, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        # the initial shape of x is: (1, 3, 32, 32)
        main_path_output = self.main_path(x)  # output tensor shape of this layer should be (1, 32, 32, 32)
        branch_path_output = self.branch_path(x)  # output tensor shape: (1, 16, 32, 32)
        combined_features = torch.cat((main_path_output, branch_path_output), dim=1)
        flattened_features = self.flatten(combined_features)
        output = self.fc1(flattened_features)
        output = self.fc2(output)
        return output

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 3, 32, 32)  # Batch size 1, 3 channels, 32x32 image
    output_tensor = model(input_tensor)
    print(output_tensor.shape)