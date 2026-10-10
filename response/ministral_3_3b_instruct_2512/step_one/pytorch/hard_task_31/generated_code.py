import torch
import torch.nn as nn

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.dropout1 = nn.Dropout(0.2)
        self.conv2 = nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1)

        self.main_path = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.Dropout(0.2)
        )
        self.branch_path = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.dropout1(x)
        x = self.conv2(x)

        main_path_output = self.main_path(x)
        branch_path_output = self.branch_path(x)
        output = torch.add(main_path_output, branch_path_output)

        return output

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 3, 32, 32)
    output_tensor = model(input_tensor)
    print(output_tensor.shape)