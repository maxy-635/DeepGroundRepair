import torch
import torch.nn as nn

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Main path
        self.main_path = nn.Sequential(
            nn.Conv1d(64, 32, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv1d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )

        # Branch path
        self.branch_path = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )

        # Combine paths
        self.combine = nn.Sequential(
            nn.Add(),
            nn.Flatten(),
            nn.Linear(32, 10)  # Assuming 10 output classes
        )

    def forward(self, x):
        main_output = self.main_path(x)
        branch_output = self.branch_path(x)

        combined_output = torch.cat((main_output, branch_output), dim=1)
        output = self.combine(combined_output)

        return output

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(32, 32, 64)
    output_tensor = model(input_tensor)
    print(output_tensor.shape)