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

        # Branch pathway
        self.branch_conv = nn.Conv2D(in_channels=1, out_channels=32, kernel_size=1, stride=1, padding=0)

        # Final layers
        self.global_avg_pool = nn.AdaptiveAvgPool2D(output_size=1)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(in_features=32, out_features=10)

    def forward(self, x):
        # Main pathway
        main = self.conv1(x)
        main = self.conv2(main)
        main = self.conv3(main)
        main = self.maxpool(main)
        main = self.dropout(main)

        # Branch pathway
        branch = self.branch_conv(x)

        # Fusion
        fused = paddle.add(main, branch)

        # Final layers
        fused = self.global_avg_pool(fused)
        fused = self.flatten(fused)
        output = self.fc(fused)

        return output

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn(shape=[1, 1, 28, 28])  # MNIST image shape: (1, 28, 28)
    output_tensor = model(input_tensor)
    print(output_tensor.shape)  # Expected output shape: [1, 10]