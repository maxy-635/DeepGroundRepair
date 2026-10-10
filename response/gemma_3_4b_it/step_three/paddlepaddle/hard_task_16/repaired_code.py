import paddle
import paddle.nn as nn

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Block 1
        self.block1_conv1 = nn.Conv2D(in_channels=3, out_channels=16, kernel_size=1, stride=1, padding=0)
        self.block1_conv2 = nn.Conv2D(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.block1_conv3 = nn.Conv2D(in_channels=32, out_channels=16, kernel_size=1, stride=1, padding=0)
        self.block1_concat = nn.Concat(dim=0)

        # Transition Convolution
        self.transition_conv = nn.Conv2D(in_channels=16, out_channels=16, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2D(num_features=16)
        self.relu1 = nn.ReLU()

        # Block 2
        self.block2_pool = nn.MaxPool2D(kernel_size=2, stride=2, padding=0)
        self.block2_fc1 = nn.Linear(16 * 8 * 8, 128)
        self.bn2 = nn.BatchNorm2D(num_features=128)
        self.relu2 = nn.ReLU()
        self.block2_fc2 = nn.Linear(128, 10)

    def forward(self, x):
        # Block 1
        x = self.block1_conv1(x)
        x = self.block1_conv2(x)
        x = self.block1_conv3(x)
        x = self.block1_concat(x)

        # Transition Convolution
        x = self.transition_conv(x)
        x = self.bn1(x)
        x = self.relu1(x)

        # Block 2
        x = self.block2_pool(x)
        x = x.reshape((x.shape[0], -1))
        x = self.block2_fc1(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.block2_fc2(x)

        return x

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn(shape=(1, 3, 32, 32))
    output_tensor = model(input_tensor)
    print(output_tensor.shape)