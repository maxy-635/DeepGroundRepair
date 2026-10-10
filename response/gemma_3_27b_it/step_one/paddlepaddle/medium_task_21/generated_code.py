import paddle
import paddle.nn as nn
import paddle.nn.functional as F

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Branch 1: 1x1 convolution
        self.branch1 = nn.Sequential(
            nn.Conv2D(in_channels=3, out_channels=32, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.Dropout(p=0.5)
        )

        # Branch 2: 1x1 convolution followed by 3x3 convolution
        self.branch2 = nn.Sequential(
            nn.Conv2D(in_channels=3, out_channels=32, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2D(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Dropout(p=0.5)
        )

        # Branch 3: 1x1 convolution followed by two consecutive 3x3 convolutions
        self.branch3 = nn.Sequential(
            nn.Conv2D(in_channels=3, out_channels=32, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2D(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2D(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Dropout(p=0.5)
        )

        # Branch 4: Average pooling followed by 1x1 convolution
        self.branch4 = nn.Sequential(
            nn.AvgPool2D(kernel_size=3, stride=1, padding=1),
            nn.Conv2D(in_channels=3, out_channels=32, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.Dropout(p=0.5)
        )

        # Fully connected layers
        self.fc1 = nn.Linear(in_features=32 * 9 * 9, out_features=128)  # Assuming input image size is 32x32
        self.fc2 = nn.Linear(in_features=128, out_features=64)
        self.fc3 = nn.Linear(in_features=64, out_features=10)

    def forward(self, x):
        # Process input through each branch
        out1 = self.branch1(x)
        out2 = self.branch2(x)
        out3 = self.branch3(x)
        out4 = self.branch4(x)

        # Concatenate the outputs from all branches
        out = paddle.concat([out1, out2, out3, out4], axis=1)

        # Flatten the concatenated output
        out = paddle.reshape(out, [out.shape[0], 32 * 9 * 9])

        # Pass through fully connected layers
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)

        return out


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn([1, 3, 32, 32])  # Batch size of 1, 3 channels, 32x32 image
    output_tensor = model(input_tensor)
    print(output_tensor.shape)