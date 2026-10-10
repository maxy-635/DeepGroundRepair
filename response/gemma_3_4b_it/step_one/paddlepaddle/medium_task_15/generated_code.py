import paddle
import paddle.nn as nn

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Convolutional layer
        self.conv1 = nn.Conv2D(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        # Batch Normalization
        self.bn1 = nn.BatchNormalization()
        # ReLU activation
        self.relu1 = nn.ReLU()
        # Global Average Pooling
        self.avgpool = nn.AvgPool2D(kernel_size=8, stride=1)
        # Fully connected layer 1
        self.fc1 = nn.Linear(16 * 8 * 8, 128)
        # ReLU activation
        self.relu2 = nn.ReLU()
        # Fully connected layer 2
        self.fc2 = nn.Linear(128, 10)
        # Reshape layer
        self.flatten = nn.Flatten()

    def forward(self, x):
        # Convolutional layer
        x = self.conv1(x)
        # Batch Normalization
        x = self.bn1(x)
        # ReLU activation
        x = self.relu1(x)
        # Global Average Pooling
        x = self.avgpool(x)
        # Flatten
        x = self.flatten(x)
        # Fully connected layer 1
        x = self.fc1(x)
        # ReLU activation
        x = self.relu2(x)
        # Fully connected layer 2
        x = self.fc2(x)
        return x

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn(shape=(1, 3, 32, 32))  # Batch size 1, 3 channels, 32x32 image
    output_tensor = model(input_tensor)
    print("Output shape:", output_tensor.shape)