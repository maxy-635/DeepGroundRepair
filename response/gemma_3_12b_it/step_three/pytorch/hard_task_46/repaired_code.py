import torch
import torch.nn as nn

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Block 1: Separable Convolutions with different kernel sizes
        self.block1_conv1x1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))
        self.block1_conv3x3 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.block1_conv5x5 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
        self.relu1 = nn.ReLU(inplace=True)
        self.cat1 = torch.cat

        # Block 2: Multiple Branches
        self.block2_conv3x3 = nn.Conv2d(in_channels=192, out_channels=128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.block2_branch1_conv1x1 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))
        self.block2_branch1_conv3x3_1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.block2_branch1_conv3x3_2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.block2_branch2_maxpool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=(0, 0))
        self.relu2 = nn.ReLU(inplace=True)
        self.cat2 = torch.cat

        # Final Layers
        self.adaptive_avg_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(in_features=128, out_features=10)  # 10 classes for CIFAR-10

    def forward(self, x):
        # Block 1
        x1_conv1x1 = self.block1_conv1x1(x)
        x1_conv3x3 = self.block1_conv3x3(x)
        x1_conv5x5 = self.block1_conv5x5(x)
        x1_relu = self.relu1(x1_conv1x1 + x1_conv3x3 + x1_conv5x5)
        x1_cat = self.cat1([x1_conv1x1, x1_conv3x3, x1_conv5x5], dim=1)

        # Block 2
        x2_conv3x3 = self.block2_conv3x3(x1_cat)
        x2_branch1_conv1x1 = self.block2_branch1_conv1x1(x2_conv3x3)
        x2_branch1_conv3x3_1 = self.block2_branch1_conv3x3_1(x2_branch1_conv1x1)
        x2_branch1_conv3x3_2 = self.block2_branch1_conv3x3_2(x2_branch1_conv3x3_1)
        x2_branch2_maxpool = self.block2_branch2_maxpool(x2_conv3x3)
        x2_relu = self.relu2(x2_branch1_conv3x3_2)
        x2_cat = self.cat2([x2_conv3x3, x2_branch1_conv3x3_2, x2_branch2_maxpool], dim=1)

        # Final Layers
        x_adaptive_avg_pool = self.adaptive_avg_pool(x2_cat)
        x_flatten = self.flatten(x_adaptive_avg_pool)
        x_linear = self.linear(x_flatten)

        return x_linear


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 3, 32, 32)  # Batch size 1, 3 channels, 32x32 image
    output_tensor = model(input_tensor)
    print(output_tensor.shape)