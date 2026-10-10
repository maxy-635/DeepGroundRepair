import paddle
import paddle.nn as nn
import paddle.vision.datasets as datasets
import paddle.vision.transforms as transforms
import numpy as np

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Path 1
        self.path1 = nn.Conv2D(in_channels=3, out_channels=32, kernel_size=1, stride=1, padding=0)
        self.relu1 = nn.ReLU()

        # Path 2
        self.avgpool2 = nn.AvgPool2D(kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2D(in_channels=3, out_channels=32, kernel_size=1, stride=1, padding=0)
        self.relu2 = nn.ReLU()

        # Path 3
        self.conv3_1 = nn.Conv2D(in_channels=3, out_channels=32, kernel_size=1, stride=1, padding=0)
        self.relu3_1 = nn.ReLU()
        self.conv3_2a = nn.Conv2D(in_channels=32, out_channels=32, kernel_size=(1, 3), stride=1, padding=0)
        self.conv3_2b = nn.Conv2D(in_channels=32, out_channels=32, kernel_size=(3, 1), stride=1, padding=0)
        self.relu3_2a = nn.ReLU()
        self.relu3_2b = nn.ReLU()

        # Path 4
        self.conv4_1 = nn.Conv2D(in_channels=3, out_channels=32, kernel_size=1, stride=1, padding=0)
        self.relu4_1 = nn.ReLU()
        self.conv4_2 = nn.Conv2D(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.relu4_2 = nn.ReLU()
        self.conv4_3a = nn.Conv2D(in_channels=32, out_channels=32, kernel_size=(1, 3), stride=1, padding=0)
        self.conv4_3b = nn.Conv2D(in_channels=32, out_channels=32, kernel_size=(3, 1), stride=1, padding=0)
        self.relu4_3a = nn.ReLU()
        self.relu4_3b = nn.ReLU()
        
        # Fully connected layer
        self.fc = nn.Linear(in_features=32 * 32 * 32, out_features=10)
        self.softmax = nn.Softmax(axis=1)

    def forward(self, x):
        # Path 1
        path1 = self.relu1(self.path1(x))

        # Path 2
        path2 = self.relu2(self.conv2(self.avgpool2(x)))

        # Path 3
        path3_1 = self.relu3_1(self.conv3_1(x))
        path3_2a = self.relu3_2a(self.conv3_2a(path3_1))
        path3_2b = self.relu3_2b(self.conv3_2b(path3_1))
        path3 = paddle.concat([path3_2a, path3_2b], axis=1)

        # Path 4
        path4_1 = self.relu4_1(self.conv4_1(x))
        path4_2 = self.relu4_2(self.conv4_2(path4_1))
        path4_3a = self.relu4_3a(self.conv4_3a(path4_2))
        path4_3b = self.relu4_3b(self.conv4_3b(path4_2))
        path4 = paddle.concat([path4_3a, path4_3b], axis=1)

        # Feature fusion
        fused_features = paddle.concat([path1, path2, path3, path4], axis=1)

        # Flatten and classify
        flattened = paddle.reshape(fused_features, [fused_features.shape[0], -1])
        output = self.softmax(self.fc(flattened))

        return output


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn([1, 3, 32, 32])
    output_tensor = model(input_tensor)
    print(output_tensor.shape)