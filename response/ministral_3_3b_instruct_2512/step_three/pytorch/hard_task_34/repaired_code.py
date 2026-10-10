import torch
import torch.nn as nn

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()
        self.main_path = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        )
        self.branch_path = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        )
        self.fuse = nn.Add()(main_path_output, branch_path_output)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(128 * 7 * 7, 10)

    def forward(self, x):
        main_path_output = self.main_path(x)
        branch_path_output = self.branch_path(main_path_output)
        fused_output = self.fuse(main_path_output, branch_path_output)
        flattened_output = self.flatten(fused_output)
        output = self.fc(flattened_output)
        return output

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 1, 28, 28)
    output_tensor = model(input_tensor)
    print(output_tensor.shape)