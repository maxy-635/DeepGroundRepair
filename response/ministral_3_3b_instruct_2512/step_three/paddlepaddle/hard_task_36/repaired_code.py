import paddle
import paddle.nn as nn

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()
        # Main pathway
        self.conv1 = nn.Conv2D(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2D(in_channels=32, out_channels=32, kernel_size=1, stride=1, padding=0)
        self.conv3 = nn.Conv2D(in_channels=32, out_channels=32, kernel_size=1, stride=1, padding=0)
        self.maxpool = nn.MaxPool2D(kernel_size=2, stride=2, padding=0)
        self.dropout = nn.Dropout(p=0.5)
        self.adaptive_avgpool = nn.AdaptiveAvgPool2D(output_size=(1, 1))

        # Branch pathway
        self.branch_conv = nn.Conv2D(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)

        # Classification layer
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(in_features=6272, out_features=10)

    def forward(self, x):
        # the initial shape of x is: (1, 1, 28, 28)
        # Main pathway
        x_main = self.conv1(x)  # output tensor shape: (1, 32, 28, 28)
        x_main = self.conv2(x_main)  # output tensor shape: (1, 32, 28, 28)
        x_main = self.conv3(x_main)  # output tensor shape: (1, 32, 28, 28)
        x_main = self.maxpool(x_main)  # output tensor shape: (1, 32, 14, 14)
        x_main = self.dropout(x_main)  # output tensor shape: (1, 32, 14, 14)
        x_main = self.adaptive_avgpool(x_main)  # output tensor shape: (1, 32, 1, 1)

        # Branch pathway
        x_branch = self.branch_conv(x)  # output tensor shape: (1, 32, 28, 28)
        x_branch = self.maxpool(x_branch)  # output tensor shape: (1, 32, 14, 14)

        # Fusion
        x_fused = paddle.add(x_main, x_branch)  # output tensor shape: (1, 32, 14, 14)

        # Classification
        x_fused = self.flatten(x_fused)  # output tensor shape: (1, 6272)
        x_output = self.fc(x_fused)

        return x_output

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.to_tensor([[[[1.0]*28]*28]*1], dtype=paddle.float32)  # Dummy input with shape [1, 1, 28, 28]
    output_tensor = model(input_tensor)
    print("Input shape:", input_tensor.shape)
    print("Output shape:", output_tensor.shape)