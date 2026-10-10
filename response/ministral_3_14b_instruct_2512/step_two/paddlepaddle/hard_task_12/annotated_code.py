import paddle
import paddle.nn as nn
import paddle.nn.functional as F

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Main path
        self.main_path_conv1x1 = nn.Conv2D(in_channels=64, out_channels=32, kernel_size=1, stride=1, padding=0)
        self.main_path_conv1x1_1 = nn.Conv2D(in_channels=32, out_channels=32, kernel_size=1, stride=1, padding=0)
        self.main_path_conv3x3 = nn.Conv2D(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)

        # Branch path
        self.branch_path_conv3x3 = nn.Conv2D(in_channels=64, out_channels='mask', kernel_size=3, stride=1, padding=1)

        # Classification layers
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_features=32*32*32, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=10)  # Assuming 10 classes for classification

    def forward(self, x):
        # the initial shape of x is: (1, 64, 32, 32)
        # Main path
        main_path = self.main_path_conv1x1(x)  # output tensor shape: (1, 32, 32, 32)
        main_path_1x1 = self.main_path_conv1x1_1(main_path)  # output tensor shape: (1, 32, 32, 32)
        main_path_3x3 = self.main_path_conv3x3(main_path)  # output tensor shape: (1, 32, 32, 32)
        main_path_output = paddle.concat([main_path_1x1, main_path_3x3], axis=1)  # output tensor shape: (1, 64, 32, 32)

        # Branch path
        branch_path_output = self.branch_path_conv3x3(x)  # output tensor shape of this layer should be (1, 64, 32, 32)

        # Combine main and branch paths
        combined_output = main_path_output + branch_path_output

        # Classification
        flattened = self.flatten(combined_output)
        fc1_output = self.fc1(flattened)
        fc2_output = self.fc2(fc1_output)

        return fc2_output

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn(shape=(1, 64, 32, 32))  # Batch size 1, 64 channels, 32x32 image
    output_tensor = model(input_tensor)
    print(output_tensor.shape)