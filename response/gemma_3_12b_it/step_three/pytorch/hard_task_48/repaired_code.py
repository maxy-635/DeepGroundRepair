import torch
import torch.nn as nn
import torch.nn.functional as F

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Block 1
        self.block1_group1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
            nn.BatchNorm2d(num_features=64)
        )
        self.block1_group2 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(num_features=64)
        )
        self.block1_group3 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2)),
            nn.BatchNorm2d(num_features=64)
        )
        self.block1_concat = nn.Sequential(
            nn.Conv2d(in_channels=192, out_channels=128, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))
        )


        # Block 2
        self.path1 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))
        )
        self.path2 = nn.Sequential(
            nn.AvgPool2d(kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        )
        self.path3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1)),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0)),
        )
        self.path4 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1)),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0)),
        )
        self.block2_concat = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))
        )
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(in_features=128 * 32 * 32, out_features=10)  # Assuming input size 32x32

    def forward(self, x):
        # the initial shape of x is: (1, 3, 32, 32)
        # Block 1
        group1 = self.block1_group1(x)  # output tensor shape: (1, 64, 32, 32)
        group2 = self.block1_group2(x)  # output tensor shape: (1, 64, 32, 32)
        group3 = self.block1_group3(x)  # output tensor shape: (1, 64, 32, 32)
        merged = torch.cat((group1, group2, group3), dim=1)  # output tensor shape: (1, 192, 32, 32)
        block1_out = self.block1_concat(merged)  # output tensor shape: (1, 128, 32, 32)

        # Block 2
        path1 = self.path1(block1_out)  # output tensor shape: (1, 64, 32, 32)
        path2 = self.path2(block1_out)
        path3 = self.path3(block1_out)
        path4 = self.path4(block1_out)
        merged2 = torch.cat((path1, path2, path3, path4), dim=1)
        block2_out = self.block2_concat(merged2)

        # Final layers
        flattened = self.flatten(block2_out)
        output = self.fc(flattened)
        return output


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 3, 32, 32)  # Batch size 1, 3 channels, 32x32 image
    output_tensor = model(input_tensor)
    print(output_tensor.shape)