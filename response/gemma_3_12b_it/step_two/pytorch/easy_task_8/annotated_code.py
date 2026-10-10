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
        self.linear = nn.Linear(in_features='mask', out_features=10)
        

    def forward(self, x):
        # the initial shape of x is: (1, 1, 28, 28)
        x = self.depthwise_conv(x)  # output tensor shape: (1, 32, 28, 28)
        x = self.depthwise_relu(x)  # output tensor shape: (1, 32, 28, 28)
        x = self.depthwise_dropout(x)  # output tensor shape: (1, 32, 28, 28)
        x = self.pointwise_conv(x)  # output tensor shape: (1, 64, 28, 28)
        x = self.pointwise_relu(x)  # output tensor shape: (1, 64, 28, 28)
        x = self.pointwise_dropout(x)  # output tensor shape: (1, 64, 28, 28)
        x = self.flatten(x)  # output tensor shape: (1, 50176)
        x = self.linear(x)
        return x


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 1, 28, 28)  # Batch size 1, 1 channel, 28x28 image
    output_tensor = model(input_tensor)
    print(output_tensor.shape)