import torch
import torch.nn as nn
import torch.nn.functional as F

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Main path
        self.main_path = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        )

        # Branch path
        self.branch_path = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=1),
            nn.Linear(in_features='mask', out_features=128, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=128, out_features=256, bias=True),
            nn.ReLU(inplace=True)
        )

        # Fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Linear(in_features=256, out_features=128, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=128, out_features=10, bias=True)
        )

    def forward(self, x):
        # the initial shape of x is: (1, 3, 32, 32)
        # Main path
        main_out = self.main_path(x)  # output tensor shape: (1, 256, 16, 16)

        # Branch path
        branch_out = self.branch_path(main_out)
        branch_out = branch_out.view(-1, 256, 1, 1)
        main_out = torch.mul(main_out, branch_out)

        # Flatten and pass through FC layers
        out = main_out.view(main_out.size(0), -1)
        out = self.fc_layers(out)

        return out

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 3, 32, 32)  # Batch size of 1, 3 channels, 32x32 image
    output_tensor = model(input_tensor)
    print(model)
    print("Input shape:", input_tensor.shape)
    print("Output shape:", output_tensor.shape)