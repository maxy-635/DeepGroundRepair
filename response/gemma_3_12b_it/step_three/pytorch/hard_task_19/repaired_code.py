import torch
import torch.nn as nn
import torch.nn.functional as F

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()
        self.main_path = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        )

        self.branch_path = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=1),
            nn.Linear(in_features=256, out_features=128),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=128, out_features=64),
            nn.ReLU(inplace=True)
        )

        self.classifier = nn.Sequential(
            nn.Linear(in_features=256 + 64, out_features=128),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=128, out_features=10)
        )

    def forward(self, x):
        # the initial shape of x is: (1, 3, 32, 32)
        main_path_output = self.main_path(x)  # output tensor shape: (1, 256, 16, 16)
        branch_path_output = self.branch_path(x)

        # Reshape branch path output to match main path output
        branch_path_output = branch_path_output.view(x.size(0), -1)
        
        # Multiply branch path output with reshaped weights
        weighted_branch_path_output = torch.mul(branch_path_output, torch.ones_like(branch_path_output))

        # Concatenate outputs from both paths
        combined_output = torch.cat((main_path_output.view(x.size(0), -1), weighted_branch_path_output), dim=1)

        # Pass through classifier
        classifier_output = self.classifier(combined_output)

        return classifier_output


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 3, 32, 32)  # Batch size 1, 3 channels, 32x32 image
    output_tensor = model(input_tensor)
    print(output_tensor.shape)