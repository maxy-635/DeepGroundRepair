import paddle
import paddle.nn as nn
import paddle.nn.functional as F

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Main path
        self.main_conv1 = nn.Conv2D(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.main_relu1 = nn.ReLU()
        self.main_conv2 = nn.Conv2D(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.main_relu2 = nn.ReLU()
        self.main_pool = nn.MaxPool2D(kernel_size=2, stride=2, padding=0)

        # Branch path
        self.branch_conv = nn.Conv2D(in_channels=1, out_channels=32, kernel_size=1, stride=1, padding=0)

        # Final layers
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(in_features=32*14*14, out_features=10)  # MNIST images are 28x28, after two max poolings they become 14x14

    def forward(self, x):
        # Main path
        main_out = self.main_conv1(x)
        main_out = self.main_relu1(main_out)
        main_out = self.main_conv2(main_out)
        main_out = self.main_relu2(main_out)
        main_out = self.main_pool(main_out)

        # Branch path
        branch_out = self.branch_conv(x)

        # Sum outputs
        sum_out = main_out + branch_out

        # Final layers
        sum_out = self.flatten(sum_out)
        out = self.fc(sum_out)

        return out

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn([1, 1, 28, 28])  # Batch of 1 MNIST image (1 channel, 28x28)
    output_tensor = model(input_tensor)
    print("Output shape:", output_tensor.shape)