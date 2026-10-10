import paddle
import paddle.nn as nn
import paddle.nn.functional as F

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Initial 1x1 convolutional layer
        self.conv1 = nn.Conv2D(in_channels=3, out_channels=16, kernel_size=1, stride=1, padding=0)
        self.relu1 = nn.ReLU()

        # Branch 1: Local Feature Extraction
        self.branch1_conv = nn.Conv2D(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.branch1_relu = nn.ReLU()

        # Branch 2: Downsampling + Convolution + Upsampling
        self.branch2_pool = nn.MaxPool2D(kernel_size=2, stride=2)
        self.branch2_conv = nn.Conv2D(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.branch2_relu = nn.ReLU()
        self.branch2_upsample = nn.Upsample(scale_factor=2, mode='nearest')

        # Branch 3: Downsampling + Convolution + Upsampling
        self.branch3_pool = nn.MaxPool2D(kernel_size=2, stride=2)
        self.branch3_conv = nn.Conv2D(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.branch3_relu = nn.ReLU()
        self.branch3_upsample = nn.Upsample(scale_factor=2, mode='nearest')

        # Fusion Layer
        self.fusion_conv = nn.Conv2D(in_channels=96, out_channels=64, kernel_size=1, stride=1, padding=0)
        self.fusion_relu = nn.ReLU()

        # Fully Connected Layers
        self.fc1 = nn.Linear(in_features=64 * 32 * 32, out_features=128)
        self.fc1_relu = nn.ReLU()
        self.fc2 = nn.Linear(in_features=128, out_features=64)
        self.fc2_relu = nn.ReLU()
        self.fc3 = nn.Linear(in_features=64, out_features=10)

    def forward(self, x):
        # Initial Convolution
        x = self.conv1(x)
        x = self.relu1(x)

        # Branch 1
        branch1 = self.branch1_conv(x)
        branch1 = self.branch1_relu(branch1)

        # Branch 2
        branch2 = self.branch2_pool(x)
        branch2 = self.branch2_conv(branch2)
        branch2 = self.branch2_relu(branch2)
        branch2 = self.branch2_upsample(branch2)

        # Branch 3
        branch3 = self.branch3_pool(x)
        branch3 = self.branch3_conv(branch3)
        branch3 = self.branch3_relu(branch3)
        branch3 = self.branch3_upsample(branch3)

        # Fusion
        x = paddle.concat([branch1, branch2, branch3], axis=1)
        x = self.fusion_conv(x)
        x = self.fusion_relu(x)

        # Flatten
        x = x.reshape([x.shape[0], -1])

        # Fully Connected Layers
        x = self.fc1(x)
        x = self.fc1_relu(x)
        x = self.fc2(x)
        x = self.fc2_relu(x)
        x = self.fc3(x)

        return x


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn(shape=[1, 3, 32, 32])
    output_tensor = model(input_tensor)
    print(output_tensor.shape)