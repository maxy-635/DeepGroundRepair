import paddle
import paddle.nn as nn

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()

        # First feature extraction block
        self.conv1_1 = nn.Conv2D(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.relu1_1 = nn.ReLU()
        self.conv1_2 = nn.Conv2D(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.relu1_2 = nn.ReLU()
        self.conv1_3 = nn.Conv2D(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.relu1_3 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2D(kernel_size=2, stride=2)

        # Second feature extraction block
        self.conv2_1 = nn.Conv2D(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.relu2_1 = nn.ReLU()
        self.conv2_2 = nn.Conv2D(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.relu2_2 = nn.ReLU()
        self.conv2_3 = nn.Conv2D(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.relu2_3 = nn.ReLU()
        self.conv2_4 = nn.Conv2D(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.relu2_4 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2D(kernel_size=2, stride=2)

        # Fully connected layers
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_features=128 * 7 * 7, out_features=512)
        self.relu_fc1 = nn.ReLU()
        self.fc2 = nn.Linear(in_features=512, out_features=256)
        self.relu_fc2 = nn.ReLU()
        self.fc3 = nn.Linear(in_features=256, out_features=10)


    def forward(self, x):
        # First feature extraction block
        x = self.conv1_1(x)
        x = self.relu1_1(x)
        x = self.conv1_2(x)
        x = self.relu1_2(x)
        x = self.conv1_3(x)
        x = self.relu1_3(x)
        x = self.maxpool1(x)

        # Second feature extraction block
        x = self.conv2_1(x)
        x = self.relu2_1(x)
        x = self.conv2_2(x)
        x = self.relu2_2(x)
        x = self.conv2_3(x)
        x = self.relu2_3(x)
        x = self.conv2_4(x)
        x = self.relu2_4(x)
        x = self.maxpool2(x)

        # Fully connected layers
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu_fc1(x)
        x = self.fc2(x)
        x = self.relu_fc2(x)
        x = self.fc3(x)

        return x


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn([1, 1, 28, 28])
    output_tensor = model(input_tensor)
    print(output_tensor.shape)