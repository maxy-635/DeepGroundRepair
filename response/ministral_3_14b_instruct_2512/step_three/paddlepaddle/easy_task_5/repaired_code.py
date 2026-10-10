import paddle
import paddle.nn as nn
import paddle.nn.functional as F

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()
        # 1x1 convolution to reduce dimensionality
        self.conv1x1_reduce = nn.Conv2D(in_channels=1, out_channels=32, kernel_size=1, stride=1, padding=0)
        # 3x3 convolution to extract features
        self.conv3x3_extract = nn.Conv2D(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        # 1x1 convolution to restore dimensionality
        self.conv1x1_restore = nn.Conv2D(in_channels=64, out_channels=32, kernel_size=1, stride=1, padding=0)
        # Flatten layer
        self.flatten = nn.Flatten()
        # Fully connected layer for classification
        self.fc = nn.Linear(in_features=32*28*28, out_features=10)

    def forward(self, x):
        x = self.conv1x1_reduce(x)
        x = F.relu(x)
        x = self.conv3x3_extract(x)
        x = F.relu(x)
        x = self.conv1x1_restore(x)  # Fixed typo here
        x = F.relu(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn([1, 1, 28, 28])  # batch_size=1, channels=1, height=28, width=28
    output_tensor = model(input_tensor)
    print(output_tensor.shape)