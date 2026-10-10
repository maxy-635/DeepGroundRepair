import paddle
import paddle.nn as nn
import paddle.nn.functional as F

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Main pathway
        self.conv3x3 = nn.Conv2D(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv1x1_1 = nn.Conv2D(in_channels=32, out_channels=16, kernel_size=1, stride=1, padding=0)
        self.conv1x1_2 = nn.Conv2D(in_channels=16, out_channels=32, kernel_size=1, stride=1, padding=0)
        self.max_pool = nn.MaxPool2D(kernel_size=2, stride=2, padding=0)
        self.dropout = nn.Dropout(p=0.5)

        # Branch pathway
        self.branch_conv = nn.Conv2D(in_channels=1, out_channels=32, kernel_size='mask', stride='mask', padding='mask')

        # Final layers
        self.global_avg_pool = nn.AdaptiveAvgPool2D(output_size=(1, 1))
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(in_features=32, out_features=10)

    def forward(self, x):
        # the initial shape of x is: (1, 1, 28, 28)
        # Main pathway
        main_out = self.conv3x3(x)  # output tensor shape: (1, 32, 28, 28)
        main_out = self.conv1x1_1(main_out)  # output tensor shape: (1, 16, 28, 28)
        main_out = self.conv1x1_2(main_out)  # output tensor shape: (1, 32, 28, 28)
        main_out = self.max_pool(main_out)  # output tensor shape: (1, 32, 14, 14)
        main_out = self.dropout(main_out)  # output tensor shape: (1, 32, 14, 14)

        # Branch pathway
        branch_out = self.branch_conv(x)  # output tensor shape of this layer should be (1, 32, 14, 14)

        # Fusion
        fused_out = main_out + branch_out

        # Final layers
        out = self.global_avg_pool(fused_out)
        out = self.flatten(out)
        out = self.fc(out)

        return out

if __name__ == "__main__":
    model = DL_Model()
    # MNIST images are 28x28 with 1 channel
    input_tensor = paddle.randn([1, 1, 28, 28])
    output_tensor = model(input_tensor)
    print("Input shape:", input_tensor.shape)
    print("Output shape:", output_tensor.shape)