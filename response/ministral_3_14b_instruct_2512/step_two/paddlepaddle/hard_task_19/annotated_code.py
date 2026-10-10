import paddle
import paddle.nn as nn
import paddle.nn.functional as F

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Main path
        self.main_conv1 = nn.Conv2D(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.main_conv2 = nn.Conv2D(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.main_conv3 = nn.Conv2D(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.main_pool = nn.MaxPool2D(kernel_size=2, stride=2, padding=0)

        # Branch path
        self.branch_pool = nn.AdaptiveAvgPool2D(output_size=(1, 1))
        self.branch_fc1 = nn.Linear(in_features=256, out_features=128)
        self.branch_fc2 = nn.Linear(in_features=128, out_features=256)

        # Final layers
        self.final_fc1 = nn.Linear(in_features='mask', out_features=128)
        self.final_fc2 = nn.Linear(in_features=128, out_features=10)

    def forward(self, x):
        # the initial shape of x is: (1, 3, 32, 32)
        # Main path
        x_main = F.relu(self.main_conv1(x))  # output tensor shape: (1, 64, 32, 32)
        x_main = F.relu(self.main_conv2(x_main))  # output tensor shape: (1, 128, 32, 32)
        x_main = F.relu(self.main_conv3(x_main))  # output tensor shape: (1, 256, 32, 32)
        x_main = self.main_pool(x_main)  # output tensor shape: (1, 256, 16, 16)

        # Branch path
        x_branch = self.branch_pool(x_main)  # output tensor shape: (1, 256, 1, 1)
        x_branch = x_branch.flatten(1)  # output tensor shape: (1, 256)
        x_branch = F.relu(self.branch_fc1(x_branch))  # output tensor shape: (1, 128)
        x_branch = F.relu(self.branch_fc2(x_branch))  # output tensor shape: (1, 256)
        x_branch = x_branch.unsqueeze(-1).unsqueeze(-1)  # output tensor shape: (1, 256, 1, 1)

        # Channel-wise multiplication
        x_main = x_main * x_branch  # output tensor shape: (1, 256, 16, 16)

        # Flatten and final layers
        x_main = x_main.flatten(1)  # output tensor shape: (1, 65536)
        x_main = F.relu(self.final_fc1(x_main))
        x_main = self.final_fc2(x_main)

        return x_main

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn([1, 3, 32, 32])  # CIFAR-10 image shape: (batch_size, 3, 32, 32)
    output_tensor = model(input_tensor)
    print("Output shape:", output_tensor.shape)