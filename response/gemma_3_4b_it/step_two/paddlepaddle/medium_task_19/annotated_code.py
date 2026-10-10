import paddle
import paddle.nn as nn

class DL_Model(nn.Layer):
    def __init__(self):
        super().__init__()

        # Branch 1: 1x1 convolution for dimensionality reduction
        self.branch1 = nn.Conv2D(in_channels=3, out_channels=64, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2D(64)
        self.relu1 = nn.ReLU()

        # Branch 2: 1x1 convolution followed by 3x3 convolution
        self.branch2 = nn.Conv2D(in_channels=64, out_channels=32, kernel_size=1, stride=1, padding=0)
        self.bn2 = nn.BatchNorm2D(32)
        self.relu2 = nn.ReLU()
        self.branch2 = nn.Conv2D(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn2_2 = nn.BatchNorm2D(32)
        self.relu2_2 = nn.ReLU()

        # Branch 3: 1x1 convolution followed by 5x5 convolution
        self.branch3 = nn.Conv2D(in_channels=32, out_channels=64, kernel_size=1, stride=1, padding=0)
        self.bn3 = nn.BatchNorm2D(64)
        self.relu3 = nn.ReLU()
        self.branch3 = nn.Conv2D(in_channels=64, out_channels=64, kernel_size=5, stride=1, padding=2)
        self.bn3_2 = nn.BatchNorm2D(64)
        self.relu3_2 = nn.ReLU()

        # Branch 4: 3x3 max pooling followed by 1x1 convolution
        self.branch4 = nn.MaxPool2D(kernel_size=3, stride=2, padding=1)
        self.branch4 = nn.Conv2D(in_channels=64, out_channels=32, kernel_size=1, stride=1, padding=0)
        self.bn4 = nn.BatchNorm2D(32)
        self.relu4 = nn.ReLU()

        # Concatenate the outputs of the branches
        self.concat_axis = 1
        self.concat = nn.Concat(dim='mask')

        # Flatten the features
        self.flatten = nn.Flatten()

        # Fully connected layers
        self.fc1 = nn.Linear(32 * 64, 128)
        self.bn_fc1 = nn.BatchNorm2D(128)
        self.relu_fc1 = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        # Branch 1
        x = self.branch1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        # Branch 2
        x = self.branch2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.branch2(x)
        x = self.bn2_2(x)
        x = self.relu2_2(x)

        # Branch 3
        x = self.branch3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.branch3(x)
        x = self.bn3_2(x)
        x = self.relu3_2(x)

        # Branch 4
        x = self.branch4(x)

        # Concatenate the outputs
        x = self.concat([x, x, x, x])

        # Flatten the features
        x = self.flatten(x)

        # Fully connected layers
        x = self.fc1(x)
        x = self.bn_fc1(x)
        x = self.relu_fc1(x)
        x = self.fc2(x)

        return x

if __name__ == "__main__":
    model = DL_Model()
    # Dummy input tensor
    input_tensor = paddle.randn(batch_size=1, channels=3, height=32, width=32)
    output_tensor = model(input_tensor)
    print(output_tensor.shape)