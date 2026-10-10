import paddle
import paddle.nn as nn
import paddle.vision.datasets as datasets
import paddle.vision.transforms as transforms

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Block 1: Separable Convolution
        self.conv1 = nn.Conv2D(in_channels=3, out_channels=64, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv2D(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2D(in_channels=64, out_channels=64, kernel_size=5, stride=1, padding=2)
        self.concat1 = nn.Concat(dim=1)

        # Block 2: Feature Extraction
        self.conv4 = nn.Conv2D(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2D(in_channels=128, out_channels=128, kernel_size=1, stride=1, padding=0)
        self.conv6 = nn.Conv2D(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2D(kernel_size=2, stride=2, padding=0)

        # Branch 1: 3x3 Conv
        self.branch1_conv = nn.Conv2D(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)

        # Branch 2: 1x1 Conv + 2x3 Convs
        self.branch2_conv1 = nn.Conv2D(in_channels=128, out_channels=128, kernel_size=1, stride=1, padding=0)
        self.branch2_conv2 = nn.Conv2D(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.branch2_conv3 = nn.Conv2D(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)

        # Branch 3: Max Pooling
        self.branch3_pool = nn.MaxPool2D(kernel_size=2, stride=2, padding=0)

        # Concatenate Branches
        self.concat2 = nn.Concat(dim=1)

        # Global Average Pooling
        self.global_avgpool = nn.AdaptiveAvgPool2D(output_size=(1, 1))

        # Fully Connected Layer
        self.fc = nn.Linear(in_features=128, out_features=10)

    def forward(self, x):
        # Block 1
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        features1 = self.concat1([x1, x2, x3])

        # Block 2
        features2 = self.conv4(features1)
        features2 = self.conv5(features2)
        features2 = self.conv6(features2)
        features2 = self.pool(features2)

        features3 = self.branch1_conv(features2)
        features4 = self.branch2_conv1(features2)
        features4 = self.branch2_conv2(features4)
        features4 = self.branch2_conv3(features4)
        features4 = self.branch3_pool(features4)
        features4 = self.concat2([features3, features4])

        # Global Average Pooling and Fully Connected Layer
        pooled = self.global_avgpool(features4)
        flattened = self.flatten(pooled)
        output = self.fc(flattened)

        return output

    def flatten(self, x):
        return paddle.flatten(x, start_dim=1)


if __name__ == "__main__":
    model = DL_Model()
    # Dummy input tensor
    input_tensor = paddle.randn(batch_size=1, channels=3, height=32, width=32)
    output_tensor = model(input_tensor)
    print(output_tensor.shape)