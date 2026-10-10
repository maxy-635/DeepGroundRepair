import paddle
import paddle.nn as nn

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Main Path
        self.conv1x1 = nn.Conv2D(in_channels=3, out_channels=32, kernel_size=1, stride=1, padding=0)
        self.conv3x3 = nn.Conv2D(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv5x5 = nn.Conv2D(in_channels=3, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.relu = nn.ReLU()

        # Branch Path
        self.branch_conv1x1 = nn.Conv2D(in_channels=3, out_channels=96, kernel_size=1, stride=1, padding=0)
        self.branch_relu = nn.ReLU()

        # Fully Connected Layers
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_features=96 * 32 * 32, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=10)


    def forward(self, x):
        # Main Path
        main_path_1x1 = self.conv1x1(x)
        main_path_1x1 = self.relu(main_path_1x1)
        main_path_3x3 = self.conv3x3(x)
        main_path_3x3 = self.relu(main_path_3x3)
        main_path_5x5 = self.conv5x5(x)
        main_path_5x5 = self.relu(main_path_5x5)

        main_path_output = paddle.concat([main_path_1x1, main_path_3x3, main_path_5x5], axis=1)

        # Branch Path
        branch_path_output = self.branch_conv1x1(x)
        branch_path_output = self.branch_relu(branch_path_output)
        
        # Add Main and Branch Paths
        x = paddle.add(main_path_output, branch_path_output)

        # Fully Connected Layers
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)

        return x


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn(shape=[1, 3, 32, 32])
    output_tensor = model(input_tensor)
    print(output_tensor.shape)