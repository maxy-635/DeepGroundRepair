import torch
import torch.nn as nn
import torch.nn.functional as F

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()
        self.main_path = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )
        self.branch_path = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        main_path_output = self.main_path(x)
        branch_path_output = self.branch_path(x)

        combined_features = torch.cat((main_path_output, branch_path_output), dim=1)
        flattened_features = combined_features.view(-1, combined_features.size(1))
        output = F.linear(flattened_features, 10)

        return output


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 1, 28, 28)  # Example input shape: batch_size, channels, height, width
    output_tensor = model(input_tensor)
    print(output_tensor.shape)