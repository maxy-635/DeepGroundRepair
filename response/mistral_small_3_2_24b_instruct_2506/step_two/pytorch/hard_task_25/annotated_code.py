import torch
import torch.nn as nn

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Main path
        self.main_conv1x1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=1, stride=1, padding=0)

        # Branch 1
        self.branch1_conv3x3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)

        # Branch 2
        self.branch2_avgpool = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        self.branch2_conv3x3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.branch2_upconv = nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=2, stride=2, padding=0)

        # Branch 3
        self.branch3_avgpool = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        self.branch3_conv3x3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.branch3_upconv = nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=2, stride=2, padding=0)

        # Concatenation and final 1x1 convolution
        self.main_conv1x1_final = nn.Conv2d(in_channels=192, out_channels=64, kernel_size=1, stride=1, padding=0)

        # Branch path
        self.branch_conv1x1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=1, stride=1, padding=0)

        # Final fully connected layer
        self.fc = nn.Linear(in_features=64*32*32, out_features=10)

    def forward(self, x):
        # Main path
        main_out = self.main_conv1x1(x)

        # Branch 1
        branch1_out = self.branch1_conv3x3(main_out)

        # Branch 2
        branch2_out = self.branch2_avgpool(main_out)
        branch2_out = self.branch2_conv3x3(branch2_out)
        branch2_out = self.branch2_upconv(branch2_out)

        # Branch 3
        branch3_out = self.branch3_avgpool(main_out)
        branch3_out = self.branch3_conv3x3(branch3_out)
        branch3_out = self.branch3_upconv(branch3_out)

        # Concatenate all branches
        concatenated = torch.cat([branch1_out, branch2_out, branch3_out], dim=1)

        # Final 1x1 convolution
        main_out = self.main_conv1x1_final(concatenated)

        # Branch path
        branch_out = self.branch_conv1x1(x)

        # Fusion
        fused_out = main_out + branch_out

        # Flatten and final classification
        fused_out = fused_out.view(fused_out.size(0), -1)
        output = self.fc(fused_out)

        return output

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 3, 32, 32)  # Batch size of 1, 3 channels, 32x32 image
    output_tensor = model(input_tensor)
    print(output_tensor.shape)  # Expected output shape: torch.Size([1, 10])