import paddle
import paddle.nn as nn

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Main path block 1
        self.main_conv1 = nn.Conv2D(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.main_relu1 = nn.ReLU()
        self.main_sep_conv1 = nn.Conv2D(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1, groups=32)
        self.main_pool1 = nn.MaxPool2D(kernel_size=2, stride=2, padding=0)

        # Main path block 2
        self.main_conv2 = nn.Conv2D(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.main_relu2 = nn.ReLU()
        self.main_sep_conv2 = nn.Conv2D(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, groups=64)
        self.main_pool2 = nn.MaxPool2D(kernel_size=2, stride=2, padding=0)

        # Branch path
        self.branch_conv = nn.Conv2D(in_channels=1, out_channels=64, kernel_size=1, stride=1, padding=0)

        # Final layers
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(in_features=64*7*7, out_features=10)

    def forward(self, x):
        # Main path
        main = self.main_conv1(x)
        main = self.main_relu1(main)
        main = self.main_sep_conv1(main)
        main = self.main_pool1(main)

        main = self.main_conv2(main)
        main = self.main_relu2(main)
        main = self.main_sep_conv2(main)
        main = self.main_pool2(main)

        # Branch path
        branch = self.branch_conv(x)

        # Sum main and branch paths
        out = main + branch

        # Final layers
        out = self.flatten(out)
        out = self.fc(out)

        return out

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn([1, 1, 28, 28])  # batch_size=1, channels=1, height=28, width=28
    output_tensor = model(input_tensor)
    print(output_tensor.shape)  # Expected output shape: [1, 10]