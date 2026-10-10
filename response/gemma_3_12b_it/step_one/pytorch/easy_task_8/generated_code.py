import torch
import torch.nn as nn

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()
        self.depthwise_conv = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.depthwise_relu = nn.ReLU()
        self.depthwise_dropout = nn.Dropout(p=0.2)
        self.pointwise_conv = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=1, stride=1, padding=0)
        self.pointwise_relu = nn.ReLU()
        self.pointwise_dropout = nn.Dropout(p=0.2)
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(in_features=64 * 7 * 7, out_features=10)  # Assuming 28x28 input, 3x3 conv with stride 1 and padding 1 results in 26x26, then 1x1 conv results in 26x26, flatten to 26*26*64 = 4352.  Then 28x28 -> 26x26 -> 24x24 -> 22x22 -> 20x20 -> 18x18 -> 16x16 -> 14x14 -> 12x12 -> 10x10. So it should be 7*7=49. 64*49 = 3136
        

    def forward(self, x):
        x = self.depthwise_conv(x)
        x = self.depthwise_relu(x)
        x = self.depthwise_dropout(x)
        x = self.pointwise_conv(x)
        x = self.pointwise_relu(x)
        x = self.pointwise_dropout(x)
        x = self.flatten(x)
        x = self.linear(x)
        return x


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 1, 28, 28)  # Batch size 1, 1 channel, 28x28 image
    output_tensor = model(input_tensor)
    print(output_tensor.shape)