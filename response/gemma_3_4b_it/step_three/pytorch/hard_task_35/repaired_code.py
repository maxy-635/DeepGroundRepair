import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )

        self.branch1 = nn.Sequential(
            nn.Linear(32 * 1 * 1, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 10)
        )
        self.branch2 = nn.Sequential(
            nn.Linear(32 * 1 * 1, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        # the initial shape of x is: (1, 3, 32, 32)
        block_output = self.block(x)  # output tensor shape: (1, 32)
        block_output = block_output * x  # output tensor shape: (1, 3, 32, 32)

        branch1_output = self.branch1(block_output)  # output tensor shape: (1, 3, 32, 10)
        branch2_output = self.branch2(block_output)  # output tensor shape: (1, 3, 32, 10)

        output = torch.cat((branch1_output, branch2_output), dim=1)  # output tensor shape: (1, 6, 32, 10)
        output = F.flatten(output)
        output = nn.Linear(output.shape[0], 10)(output)

        return output


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 3, 32, 32)  # Batch size 1, 3 channels, 32x32 image
    output_tensor = model(input_tensor)
    print(output_tensor.shape)