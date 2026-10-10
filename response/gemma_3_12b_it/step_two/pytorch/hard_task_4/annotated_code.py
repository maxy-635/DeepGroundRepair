import torch
import torch.nn as nn
import torch.nn.functional as F

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()
        self.initial_conv1 = nn.Conv2d(in_channels=3, out_channels=9, kernel_size=1, stride=1, padding=0)
        self.depthwise_conv = nn.Conv2d(in_channels=9, out_channels=9, kernel_size=3, stride=1, padding=1, groups=9)
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.fc1 = nn.Linear(in_features=9, out_features=9)
        self.fc2 = nn.Linear(in_features=9, out_features=9)
        self.final_conv = nn.Conv2d(in_channels='mask', out_channels='mask', kernel_size='mask', stride='mask', padding='mask')
        self.flatten = nn.Flatten()
        self.classifier = nn.Linear(in_features=3 * 32 * 32, out_features=10)

    def forward(self, x):
        # the initial shape of x is: (1, 3, 32, 32)
        x = self.initial_conv1(x)  # output tensor shape: (1, 9, 32, 32)
        x = self.depthwise_conv(x)  # output tensor shape: (1, 9, 32, 32)

        # Channel Attention
        attn = self.avg_pool(x)  # output tensor shape: (1, 9, 1, 1)
        attn = self.fc1(attn.view(x.size(0), -1))  # output tensor shape: (1, 9)
        attn = torch.sigmoid(self.fc2(attn.view(x.size(0), -1)))  # output tensor shape: (1, 9)
        attn = attn.view(x.size(0), 9, 1, 1)  # output tensor shape: (1, 9, 1, 1)
        x = x * attn  # output tensor shape: (1, 9, 32, 32)

        x = self.final_conv(x)
        x = x + torch.nn.functional.interpolate(x, size=(32,32), mode='bilinear', align_corners=False)
        x = self.flatten(x)
        x = self.classifier(x)
        return x

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 3, 32, 32)
    output_tensor = model(input_tensor)
    print(output_tensor.shape)