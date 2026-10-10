import paddle
import paddle.nn as nn
import paddle.nn.functional as F

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Channel group 1: 1x1 convolution
        self.conv1x1 = nn.Sequential(
            nn.Conv2D(in_channels=3, out_channels=64, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.BatchNorm2D(num_features=64),
            nn.MaxPool2D(kernel_size=2, stride=2, padding=0),
            nn.Dropout(p=0.25)
        )

        # Channel group 2: 3x3 convolution
        self.conv3x3 = nn.Sequential(
            nn.Conv2D(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2D(num_features=64),
            nn.MaxPool2D(kernel_size=2, stride=2, padding=0),
            nn.Dropout(p=0.25)
        )

        # Channel group 3: 5x5 convolution
        self.conv5x5 = nn.Sequential(
            nn.Conv2D(in_channels=3, out_channels=64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.BatchNorm2D(num_features=64),
            nn.MaxPool2D(kernel_size=2, stride=2, padding=0),
            nn.Dropout(p=0.25)
        )

        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(in_features=64 * 8 * 8 * 3, out_features=1024),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=1024, out_features=512),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=512, out_features=10)
        )

    def forward(self, x):
        # Split the input into three channel groups
        x1 = self.conv1x1(x)
        x2 = self.conv3x3(x)
        x3 = self.conv5x5(x)

        # Concatenate the outputs
        x = paddle.concat([x1, x2, x3], axis=1)

        # Flatten the concatenated features
        x = paddle.flatten(x, start_axis=1, stop_axis=-1)

        # Pass through fully connected layers
        x = self.fc(x)

        return x

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn(shape=[1, 3, 32, 32])  # CIFAR-10 image shape: 3x32x32
    output_tensor = model(input_tensor)
    print(output_tensor.shape)  # Expected output shape: [1, 10]