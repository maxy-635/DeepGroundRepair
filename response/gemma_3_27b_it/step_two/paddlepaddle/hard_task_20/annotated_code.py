import paddle
import paddle.nn as nn
import paddle.nn.functional as F

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Main Path
        self.main_path_conv1 = nn.Conv2D(in_channels=3, out_channels=64, kernel_size=1, stride=1, padding=0)
        self.main_path_relu1 = nn.ReLU()
        self.main_path_conv2 = nn.Conv2D(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.main_path_relu2 = nn.ReLU()
        self.main_path_conv3 = nn.Conv2D(in_channels=64, out_channels=64, kernel_size=5, stride=1, padding=2)
        self.main_path_relu3 = nn.ReLU()

        # Branch Path
        self.branch_path_conv = nn.Conv2D(in_channels=3, out_channels=64, kernel_size=1, stride=1, padding=0)
        self.branch_path_relu = nn.ReLU()

        # Fusion
        # Fully Connected Layers
        self.fc1 = nn.Linear(in_features='mask', out_features=128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(in_features=128, out_features=10)

    def forward(self, x):
        # the initial shape of x is: (1, 3, 32, 32)
        # Main Path
        main_path = self.main_path_conv1(x)  # output tensor shape: (1, 64, 32, 32)
        main_path = self.main_path_relu1(main_path)  # output tensor shape: (1, 64, 32, 32)
        main_path = self.main_path_conv2(main_path)  # output tensor shape: (1, 64, 32, 32)
        main_path = self.main_path_relu2(main_path)  # output tensor shape: (1, 64, 32, 32)
        main_path = self.main_path_conv3(main_path)  # output tensor shape: (1, 64, 32, 32)
        main_path = self.main_path_relu3(main_path)  # output tensor shape: (1, 64, 32, 32)

        # Branch Path
        branch_path = self.branch_path_conv(x)  # output tensor shape: (1, 64, 32, 32)
        branch_path = self.branch_path_relu(branch_path)  # output tensor shape: (1, 64, 32, 32)

        # Fusion
        fused_features = paddle.add(main_path, branch_path)  # output tensor shape: (1, 64, 32, 32)

        # Classification
        fused_features = paddle.nn.Flatten()(fused_features)  # output tensor shape: (1, 65536)
        x = self.fc1(fused_features)
        x = self.relu(x)
        x = self.fc2(x)
        return x


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn([1, 3, 32, 32])
    output_tensor = model(input_tensor)
    print(output_tensor.shape)