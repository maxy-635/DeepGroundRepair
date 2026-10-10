import torch
import torch.nn as nn
import torch.nn.functional as F

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Block 1
        self.block1_conv1_1 = nn.Conv2d(in_channels=3, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.block1_conv1_2 = nn.Conv2d(in_channels=3, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.block1_conv1_3 = nn.Conv2d(in_channels=3, out_channels=1, kernel_size=1, stride=1, padding=0)

    # Block 2
        self.block2_reshape = nn.Identity()
        self.block2_permute = nn.Identity()

    # Block 3
        self.block3_depthwise = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=1, groups=1, bias=False)
        self.block3_pointwise = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=1, stride=1, padding=0, groups=1, bias=False)

        # Branch
        self.branch_conv = nn.Conv2d(in_channels=3, out_channels=1, kernel_size=1, stride=1, padding=0)

        # Fully Connected Layer
        self.fc = nn.Linear(in_features=1, out_features=10)

    def forward(self, x):
        # the initial shape of x is: (1, 3, 32, 32)
        # Block 1
        x1_1 = self.block1_conv1_1(x)  # output tensor shape: (1, 1, 32, 32)
        x1_2 = self.block1_conv1_2(x)  # output tensor shape: (1, 1, 32, 32)
        x1_3 = self.block1_conv1_3(x)  # output tensor shape: (1, 1, 32, 32)
        x1 = torch.cat([x1_1, x1_2, x1_3], dim=1)  # output tensor shape: (1, 3, 32, 32)

        # Block 2
        batch_size, channels, height, width = x1.shape  # output tensor shape: (1); (1); (1); (1)
        x2 = x1.reshape(batch_size, 3, height, width)  # output tensor shape: (1, 3, 32, 32)
        x2 = x2.permute(0, 3, 1, 2)  # output tensor shape: (1, 32, 3, 32)
        x2 = x2.reshape(batch_size, channels, height, width)  # output tensor shape: (1, 3, 32, 32)

        # Block 3
        x3 = F.depthwise_conv2d(x2, self.block3_depthwise.weight, stride=1, padding=1, bias=self.block3_depthwise.bias)
        x3 = self.block3_pointwise(x3)

        # Branch
        branch_out = self.branch_conv(x)

        # Combine main path and branch
        combined = torch.add(x3, branch_out)

        # Flatten and pass through fully connected layer
        flattened = combined.view(-1, 1)
        output = self.fc(flattened)

        return output

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 3, 32, 32)
    output_tensor = model(input_tensor)
    print(output_tensor.shape)