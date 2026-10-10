import paddle
import paddle.nn as nn
import paddle.vision.datasets as datasets
import paddle.vision.transforms as transforms

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Branch 1: 1x1 convolution
        self.branch1 = nn.Conv2D(in_channels=3, out_channels=64, kernel_size=1, stride=1, padding=0)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(p=0.25)

        # Branch 2: 1x1 convolution followed by 3x3 convolution
        self.branch2 = nn.Conv2D(in_channels=64, out_channels=128, kernel_size=1, stride=1, padding=0)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(p=0.25)

        # Branch 3: 1x1 convolution followed by two consecutive 3x3 convolutions
        self.branch3 = nn.Conv2D(in_channels=128, out_channels=256, kernel_size=1, stride=1, padding=0)
        self.relu3 = nn.ReLU()
        self.dropout3 = nn.Dropout(p=0.25)
        self.branch3 = nn.Conv2D(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.relu3_1 = nn.ReLU()
        self.dropout3 = nn.Dropout(p=0.25)
        self.branch3 = nn.Conv2D(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.relu3_2 = nn.ReLU()
        self.dropout3 = nn.Dropout(p=0.25)

        # Branch 4: Average pooling followed by 1x1 convolution
        self.branch4 = nn.AvgPool2D(pool_size=8, stride=8)
        self.relu4 = nn.ReLU()
        self.dropout4 = nn.Dropout(p=0.25)
        self.branch4 = nn.Conv2D(in_channels=64, out_channels=256, kernel_size=1, stride=1, padding=0)
        self.relu4_1 = nn.ReLU()
        self.dropout4 = nn.Dropout(p=0.25)

        # Concatenate branches
        self.concat_bn = nn.Conv2D(in_channels=256, out_channels=512, kernel_size=1, stride=1, padding=0)
        self.relu_concat = nn.ReLU()
        self.dropout_concat = nn.Dropout(p=0.5)

        # Fully connected layers
        self.fc1 = nn.Linear(512, 512)
        self.relu_fc1 = nn.ReLU()
        self.dropout_fc1 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(512, 256)
        self.relu_fc2 = nn.ReLU()
        self.dropout_fc2 = nn.Dropout(p=0.5)
        self.fc3 = nn.Linear(256, 10)
        self.softmax = nn.Softmax(axis=-1)

    def forward(self, x):
        # Branch 1
        x1 = self.branch1(x)
        x1 = self.relu1(x1)
        x1 = self.dropout1(x1)

        # Branch 2
        x2 = self.branch2(x1)
        x2 = self.relu2(x2)
        x2 = self.dropout2(x2)

        # Branch 3
        x3 = self.branch3(x2)
        x3 = self.relu3(x3)
        x3 = self.dropout3(x3)

        # Branch 4
        x4 = self.branch4(x3)
        x4 = self.relu4(x4)
        x4 = self.dropout4(x4)

        # Concatenate
        x = paddle.concat((x1, x2, x3, x4), dim=1)
        x = self.concat_bn(x)
        x = self.relu_concat(x)
        x = self.dropout_concat(x)

        # Fully connected layers
        x = x.flatten(start_dim=1)
        x = self.fc1(x)
        x = self.relu_fc1(x)
        x = self.dropout_fc1(x)
        x = self.fc2(x)
        x = self.relu_fc2(x)
        x = self.dropout_fc2(x)
        x = self.fc3(x)
        x = self.softmax(x)

        return x


if __name__ == "__main__":
    model = DL_Model()
    # Dummy input tensor
    dummy_input = paddle.randn(shape=(1, 3, 32, 32))
    output_tensor = model(dummy_input)
    print(output_tensor.shape)