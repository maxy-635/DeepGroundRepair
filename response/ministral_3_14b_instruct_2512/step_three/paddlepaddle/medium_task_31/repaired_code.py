import paddle
import paddle.nn as nn
import paddle.nn.functional as F

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Split the channel dimension into three groups
        self.conv1x1_group = nn.Sequential(
            nn.Conv2D(in_channels=3, out_channels=64, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2D(num_features=64),
            nn.ReLU()
        )

        self.conv3x3_group = nn.Sequential(
            nn.Conv2D(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2D(num_features=64),
            nn.ReLU()
        )

        self.conv5x5_group = nn.Sequential(
            nn.Conv2D(in_channels=3, out_channels=64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2D(num_features=64),
            nn.ReLU()
        )

        # Concatenate the outputs from the three groups
        self.concat = nn.Sequential()

        # Fully connected layers
        self.fc1 = nn.Linear(in_features=196608, out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=10)
        self.flatten = nn.Flatten()

    def forward(self, x):
        # the initial shape of x is: (1, 3, 32, 32)
        # Split the input into three groups along the channel dimension
        # Since the input is already in NCHW format, we can directly apply the conv layers
        conv1x1_out = self.conv1x1_group(x)  # output tensor shape: (1, 64, 32, 32)
        conv3x3_out = self.conv3x3_group(x)  # output tensor shape: (1, 64, 32, 32)
        conv5x5_out = self.conv5x5_group(x)  # output tensor shape: (1, 64, 32, 32)

        # Concatenate along the channel dimension
        concat_out = paddle.concat([conv1x1_out, conv3x3_out, conv5x5_out], axis=1)  # output tensor shape: (1, 192, 32, 32)

        # Flatten and pass through fully connected layers
        flattened = self.flatten(concat_out)  # output tensor shape: (1, 196608)
        fc1_out = self.fc1(flattened)
        fc2_out = self.fc2(fc1_out)

        return fc2_out

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn((1, 3, 32, 32))  # Batch size of 1, 3 channels, 32x32 image
    output_tensor = model(input_tensor)
    print("Output shape:", output_tensor.shape)