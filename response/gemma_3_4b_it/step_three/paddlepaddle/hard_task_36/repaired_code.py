import paddle
import paddle.nn as nn

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Main pathway
        self.conv1 = nn.Conv2D(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2D(num_features=32)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2D(kernel_size=2, stride=2, padding=0)
        self.dropout1 = nn.Dropout(p=0.5)

        # Branch pathway
        self.conv_branch = nn.Conv2D(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn_branch = nn.BatchNorm2D(num_features=32)
        self.relu_branch = nn.ReLU()

        # Fusion
        self.add = nn.Add()

        # Global average pooling
        self.global_avg_pool = nn.AdaptiveAvgPool2D(output_size=(1, 1))
        self.flatten = nn.Flatten()

        # Fully connected layer
        self.linear = nn.Linear(in_features=32, out_features=10)  # 10 classes for MNIST

    def forward(self, x):
        # Main pathway
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.dropout1(x)

        # Branch pathway
        x_branch = self.conv_branch(x)
        x_branch = self.bn_branch(x_branch)
        x_branch = self.relu_branch(x_branch)

        # Fusion
        x = self.add(x, x_branch)

        # Global average pooling and flattening
        x = self.global_avg_pool(x)
        x = self.flatten(x)

        # Fully connected layer
        x = self.linear(x)

        # Softmax
        x = nn.Softmax(axis=-1)(x)

        return x


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn(batch_size=1, C=1, H=28, W=28)  # Batch size 1, 1 channel, 28x28 image
    output_tensor = model(input_tensor)
    print(output_tensor.shape)