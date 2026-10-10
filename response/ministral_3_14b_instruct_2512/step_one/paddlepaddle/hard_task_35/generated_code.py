import paddle
import paddle.nn as nn
import paddle.nn.functional as F

class Block(nn.Layer):
    def __init__(self, in_channels):
        super(Block, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2D(1)
        self.fc1 = nn.Linear(in_channels, in_channels // 2)
        self.fc2 = nn.Linear(in_channels // 2, in_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        # Global average pooling
        pooled = self.avg_pool(x)
        pooled = pooled.reshape([pooled.shape[0], -1])

        # Two fully connected layers
        fc_out = self.fc1(pooled)
        fc_out = self.fc2(fc_out)
        fc_out = self.relu(fc_out)

        # Reshape and element-wise multiply
        weights = fc_out.reshape([fc_out.shape[0], -1, 1, 1])
        return paddle.multiply(x, weights)

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()

        # First branch
        self.conv1 = nn.Conv2D(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2D(num_features=64)
        self.relu1 = nn.ReLU()
        self.block1 = Block(in_channels=64)

        # Second branch
        self.conv2 = nn.Conv2D(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2D(num_features=64)
        self.relu2 = nn.ReLU()
        self.block2 = Block(in_channels=64)

        # Final layers
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(in_features=64*32*32, out_features=10)  # CIFAR-10 has 10 classes

    def forward(self, x):
        # First branch
        x1 = self.conv1(x)
        x1 = self.bn1(x1)
        x1 = self.relu1(x1)
        x1 = self.block1(x1)

        # Second branch
        x2 = self.conv2(x)
        x2 = self.bn2(x2)
        x2 = self.relu2(x2)
        x2 = self.block2(x2)

        # Concatenate and final layers
        x = paddle.concat([x1, x2], axis=1)
        x = self.flatten(x)
        x = self.fc(x)
        return x

if __name__ == "__main__":
    model = DL_Model()
    # CIFAR-10 images are 32x32 with 3 channels
    input_tensor = paddle.randn([1, 3, 32, 32])
    output_tensor = model(input_tensor)
    print("Input shape:", input_tensor.shape)
    print("Output shape:", output_tensor.shape)