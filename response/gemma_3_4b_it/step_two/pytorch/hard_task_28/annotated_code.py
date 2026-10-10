import torch
import torch.nn as nn

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()
        self.main_path = nn.Sequential(
            nn.Conv2d(3, 32, 7, stride=1, padding=3),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 1, stride=1, padding=0),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 1, stride=1, padding=0),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.branch_path = nn.Sequential(
            nn.Conv2d(3, 32, 7, stride=1, padding=3)
        )
        self.combine = nn.Sequential(
            nn.Add(),
            nn.Flatten(),
            nn.Linear(32 * 32 * 32, 10)
        )

    def forward(self, x):
        main_path_output = self.main_path(x)
        branch_path_output = self.branch_path(x)
        combined_output = torch.add(main_path_output, branch_path_output)
        output_tensor = self.combine(combined_output)
        return output_tensor


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 3, 32, 32)
    output_tensor = model(input_tensor)
    print(output_tensor.shape)