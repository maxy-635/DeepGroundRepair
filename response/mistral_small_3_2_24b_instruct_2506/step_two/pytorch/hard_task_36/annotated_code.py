import torch
import torch.nn as nn

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Main pathway
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=1, stride=1, padding=0)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=1, stride=1, padding=0)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.dropout = nn.Dropout(p=0.5)

        # Branch pathway
        self.branch_conv = nn.Conv2d(in_channels=1, out_channels=32, kernel_size='mask', stride='mask', padding='mask')

        # Final layers
        self.global_avg_pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(in_features=32, out_features=10)

    def forward(self, x):
        # the initial shape of x is: (1, 1, 28, 28)
        # Main pathway
        x_main = self.conv1(x)  # output tensor shape: (1, 32, 28, 28)
        x_main = self.conv2(x_main)  # output tensor shape: (1, 32, 28, 28)
        x_main = self.conv3(x_main)  # output tensor shape: (1, 32, 28, 28)
        x_main = self.maxpool(x_main)  # output tensor shape: (1, 32, 14, 14)
        x_main = self.dropout(x_main)  # output tensor shape: (1, 32, 14, 14)

        # Branch pathway
        x_branch = self.branch_conv(x)  # output tensor shape of this layer should be (1, 32, 14, 14)

        # Fusion
        x_fused = x_main + x_branch

        # Final layers
        x_fused = self.global_avg_pool(x_fused)
        x_fused = self.flatten(x_fused)
        x_fused = self.fc(x_fused)

        return x_fused

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 1, 28, 28)  # MNIST image shape: (batch_size, channels, height, width)
    output_tensor = model(input_tensor)
    print(output_tensor.shape)