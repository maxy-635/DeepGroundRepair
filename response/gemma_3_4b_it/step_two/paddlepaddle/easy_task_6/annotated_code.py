import paddle
import paddle.vision.datasets as datasets
import paddle.nn as nn

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Branch 1
        self.conv1 = nn.Conv2D(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2D(num_features=16)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2D(in_channels=16, out_channels=8, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2D(num_features=8)
        self.relu2 = nn.ReLU()

        # Branch 2 (Direct input)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_features=8 * 28 * 28, out_features=64)
        self.relu3 = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(in_features=64, out_features=10)
        self.softmax = nn.Softmax2D()

        # Combine branches
        self.add = nn.Add()

    def forward(self, x):
        # Branch 1
        branch1 = self.conv1(x)
        branch1 = self.bn1(branch1)
        branch1 = self.relu1(branch1)
        branch1 = self.conv2(branch1)
        branch1 = self.bn2(branch1)
        branch1 = self.relu2(branch1)

        # Branch 2 (Direct input)
        branch2 = self.flatten(x)
        branch2 = self.fc1(branch2)
        branch2 = self.relu3(branch2)
        branch2 = self.dropout(branch2)
        branch2 = self.fc2(branch2)
        branch2 = self.softmax(branch2)

        # Combine branches
        output = self.add([branch1, branch2])

        return output


if __name__ == "__main__":
    model = DL_Model()
    # Dummy input tensor (batch_size=1, channels=1, height=28, width=28)
    input_tensor = paddle.randn(shape=(1, 1, 28, 28))
    output_tensor = model(input_tensor)
    print("Output shape:", output_tensor.shape)