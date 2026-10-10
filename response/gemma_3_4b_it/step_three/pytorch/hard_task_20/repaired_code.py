import torch
import torch.nn as nn

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()
        self.main_path = nn.Sequential(
            nn.Conv1d(3, 16, kernel_size=1, stride=1, padding=0),
            nn.Conv1d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.Conv1d(32, 64, kernel_size=5, stride=1, padding=2)
        )
        self.branch_path = nn.Conv1d(3, 64, kernel_size=1, stride=1, padding=0)
        self.fc1 = nn.Linear(in_features=3, out_features=64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        # the initial shape of x is: (1, 3, 32)
        main_output = self.main_path(x)  # output tensor shape: (1, 64, 32)
        branch_output = self.branch_path(x)  # output tensor shape: (1, 64, 32)
        fused_features = torch.add(main_output, branch_output)  # output tensor shape: (1, 64, 32)
        output = self.fc1(fused_features)
        output = self.fc2(output)
        return output

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 3, 32)  # Batch size 1, 3 channels, 32 spatial dimensions
    output_tensor = model(input_tensor)
    print(output_tensor.shape)