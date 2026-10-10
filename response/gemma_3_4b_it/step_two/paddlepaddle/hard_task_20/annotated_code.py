import paddle
import paddle.nn as nn

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Main path
        self.conv1 = nn.Conv2D(in_channels=3, out_channels=64, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2D(64)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2D(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2D(64)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2D(in_channels=64, out_channels=64, kernel_size=5, stride=1, padding=2)
        self.bn3 = nn.BatchNorm2D(64)
        self.relu3 = nn.ReLU()
        self.concat = nn.Concat(dim='mask')

        # Branch path
        self.conv_branch = nn.Conv2D(in_channels=3, out_channels=64, kernel_size=1, stride=1, padding=0)
        self.bn_branch = nn.BatchNorm2D(64)
        self.relu_branch = nn.ReLU()

        # Fully connected layers
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 8 * 8, 128)  # Adjust input size based on CIFAR-10 image size
        self.bn_fc1 = nn.BatchNorm2D(128)
        self.relu_fc1 = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)
        self.softmax = nn.Softmax(axis=-1)

    def forward(self, x):
        # Main path
        main_path = self.conv1(x)
        main_path = self.bn1(main_path)
        main_path = self.relu1(main_path)
        main_path = self.conv2(main_path)
        main_path = self.bn2(main_path)
        main_path = self.relu2(main_path)
        main_path = self.conv3(main_path)
        main_path = self.bn3(main_path)
        main_path = self.relu3(main_path)
        main_path = self.concat(main_path)

        # Branch path
        branch_path = self.conv_branch(x)
        branch_path = self.bn_branch(branch_path)
        branch_path = self.relu_branch(branch_path)
        branch_path = self.flatten(branch_path)
        branch_path = self.fc1(branch_path)
        branch_path = self.bn_fc1(branch_path)
        branch_path = self.relu_fc1(branch_path)

        # Combine paths
        fused_features = main_path + branch_path

        # Classification
        output = self.fc2(fused_features)
        output = self.softmax(output)

        return output


if __name__ == "__main__":
    model = DL_Model()
    # Dummy input tensor (batch_size=1, channels=3, height=32, width=32)
    input_tensor = paddle.randn(shape=(1, 3, 32, 32))
    output_tensor = model(input_tensor)
    print(output_tensor.shape)