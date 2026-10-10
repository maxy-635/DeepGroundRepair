import torch
import torch.nn as nn
import torch.nn.functional as F

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Main Path
        self.main_path_conv1x1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=1, stride=1, padding=0)
        self.main_path_split = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.AvgPool2d(kernel_size=2, stride=2, padding=0),
            nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=2, stride=2, padding=0),
            nn.AvgPool2d(kernel_size=2, stride=2, padding=0),
            nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=2, stride=2, padding=0)
        )
        self.main_path_concat = nn.Conv2d(in_channels=192, out_channels=64, kernel_size=1, stride=1, padding=0)
        
        # Branch Path
        self.branch_path_conv1x1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=1, stride=1, padding=0)

        # Fusion
        self.fusion = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=1, stride=1, padding=0)

        # Classification
        self.fc = nn.Linear(in_features=65536, out_features=10)

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(64)


    def forward(self, x):
        # the initial shape of x is: (1, 3, 32, 32)
        # Main Path
        main_path_output = self.main_path_conv1x1(x)  # output tensor shape: (1, 64, 32, 32)
        main_path_branch1 = self.main_path_split(main_path_output)  # output tensor shape: (1, 64, 32, 32)
        
        # Branch Path
        branch_path_output = self.branch_path_conv1x1(x)  # output tensor shape: (1, 64, 32, 32)

        # Concatenate and Fuse
        cat_output = torch.cat((main_path_branch1, branch_path_output), dim=1)  # output tensor shape: (1, 128, 32, 32)
        
        fused_output = self.fusion(cat_output)  # output tensor shape: (1, 64, 32, 32)

        # Flatten and Classify
        flattened_output = torch.flatten(fused_output, 1)  # output tensor shape: (1, 65536)
        output = self.fc(flattened_output)

        return output


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 3, 32, 32)  # Batch size 1, 3 channels, 32x32 image
    output_tensor = model(input_tensor)
    print(output_tensor.shape)