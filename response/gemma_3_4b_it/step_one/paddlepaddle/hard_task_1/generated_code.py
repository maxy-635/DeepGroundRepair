import paddle
import paddle.nn as nn

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Block 1
        self.conv1 = nn.Conv2D(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.avgpool1 = nn.AvgPool2D(kernel_size=2, stride=2)
        self.fc1_1 = nn.Linear(64 * 32 * 32, 256)
        self.fc1_2 = nn.Linear(64 * 32 * 32, 256)

        # Block 2
        self.avgpool2 = nn.AvgPool2D(kernel_size=2, stride=2)
        self.maxpool2 = nn.MaxPool2D(kernel_size=2, stride=2)
        self.concat = nn.Concat(dim=1)
        self.conv2_1 = nn.Conv2D(in_channels=64, out_channels=64, kernel_size=1, stride=1, padding=0)
        self.sigmoid = nn.Sigmoid()
        self.scale = nn.Linear(64 * 32 * 32, 64 * 32 * 32)

        # Final layer
        self.fc3 = nn.Linear(64 * 32 * 32, 10)

    def forward(self, x):
        # Block 1
        x = self.conv1(x)
        x = self.avgpool1(x)
        x = x.reshape(x.shape[0], -1)
        x1 = self.fc1_1(x)
        x2 = self.fc1_2(x)
        x = torch.cat((x1, x2), dim=1)

        x = self.sigmoid(x)
        x = x * x1

        # Block 2
        x = self.avgpool2(x)
        x = self.maxpool2(x)
        x = self.concat(x)
        x = self.conv2_1(x)
        x = self.sigmoid(x)
        x = x * x1

        # Final branch
        x = self.conv1(x)
        x = self.scale(x)
        x = x + x1
        x = self.fc3(x)

        return x

if __name__ == "__main__":
    import torch
    model = DL_Model()
    input_tensor = torch.randn(1, 3, 32, 32)
    output_tensor = model(input_tensor)
    print(output_tensor.shape)