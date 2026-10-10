import paddle
import paddle.nn as nn

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Block 1
        self.conv1_1 = nn.Conv2D(in_channels=3, out_channels=16, kernel_size=1, stride=1, padding=0)
        self.relu1_1 = nn.ReLU()
        self.conv1_2 = nn.Conv2D(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.relu1_2 = nn.ReLU()
        self.conv1_3 = nn.Conv2D(in_channels=16, out_channels=16, kernel_size=5, stride=1, padding=2)
        self.relu1_3 = nn.ReLU()
        self.dropout1 = nn.Dropout(p=0.5)
        self.concat1 = nn.Concat(dim=1)

        # Block 2
        self.conv2_1 = nn.Conv2D(in_channels=16, out_channels=32, kernel_size=1, stride=1, padding=0)
        self.bn2_1 = nn.BatchNorm2D(num_features=32)
        self.relu2_1 = nn.ReLU()
        self.conv2_2 = nn.Conv2D(in_channels=32, out_channels=32, kernel_size=1, stride=1, padding=0)
        self.bn2_2 = nn.BatchNorm2D(num_features=32)
        self.relu2_2 = nn.ReLU()
        self.conv2_3 = nn.Conv2D(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.relu2_3 = nn.ReLU()
        self.conv2_4 = nn.Conv2D(in_channels=32, out_channels=32, kernel_size=1, stride=1, padding=0)
        self.bn2_4 = nn.BatchNorm2D(num_features=32)
        self.relu2_4 = nn.ReLU()
        self.conv2_5 = nn.Conv2D(in_channels=32, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.relu2_5 = nn.ReLU()
        self.conv2_6 = nn.Conv2D(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.relu2_6 = nn.ReLU()
        self.concat2 = nn.Concat(dim=1)

        # Flatten and Linear
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(in_features=32 * 8, out_features=10)  # Assuming 10 classes for CIFAR-10

    def forward(self, x):
        # Block 1
        x = self.conv1_1(x)
        x = self.relu1_1(x)
        x = self.conv1_2(x)
        x = self.relu1_2(x)
        x = self.conv1_3(x)
        x = self.relu1_3(x)
        x = self.dropout1(x)
        x = self.concat1(x)

        # Block 2
        x = self.conv2_1(x)
        x = self.bn2_1(x)
        x = self.relu2_1(x)
        x = self.conv2_2(x)
        x = self.bn2_2(x)
        x = self.relu2_2(x)
        x = self.conv2_3(x)
        x = self.relu2_3(x)
        x = self.conv2_4(x)
        x = self.bn2_4(x)
        x = self.relu2_4(x)
        x = self.conv2_5(x)
        x = self.relu2_5(x)
        x = self.conv2_6(x)
        x = self.relu2_6(x)
        x = self.concat2(x)

        # Flatten and Linear
        x = self.flatten(x)
        x = self.linear(x)

        return x

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn(batch_size=4, channels=3, height=32, width=32)
    output_tensor = model(input_tensor)
    print(output_tensor.shape)