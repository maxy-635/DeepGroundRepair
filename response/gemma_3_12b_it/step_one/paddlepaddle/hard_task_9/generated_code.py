import paddle
import paddle.nn as nn
import paddle.nn.functional as F

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()
        # Feature extraction branches
        self.branch1 = nn.Sequential(
            nn.Conv2D(in_channels=3, out_channels=32, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2D(num_features=32),
            nn.ReLU()
        )
        self.branch2 = nn.Sequential(
            nn.Conv2D(in_channels=3, out_channels=32, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2D(num_features=32),
            nn.ReLU(),
            nn.Conv2D(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2D(num_features=32),
            nn.ReLU()
        )
        self.branch3 = nn.Sequential(
            nn.Conv2D(in_channels=3, out_channels=32, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2D(num_features=32),
            nn.ReLU(),
            nn.Conv2D(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2D(num_features=32),
            nn.ReLU(),
            nn.Conv2D(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2D(num_features=32),
            nn.ReLU()
        )
        
        # Concatenate the branches
        self.concat_conv = nn.Conv2D(in_channels=96, out_channels=32, kernel_size=1, stride=1, padding=0)

        # Direct connection to the input
        self.direct_connection = nn.Sequential(
            nn.BatchNorm2D(num_features=3),
            nn.ReLU()
        )

        # Fuse the main path and the branch
        self.fuse_add = paddle.add

        # Fully connected layers
        self.fc1 = nn.Linear(in_features=32 * 32 * 32, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=64)
        self.fc3 = nn.Linear(in_features=64, out_features=10)

    def forward(self, x):
        # Feature extraction
        branch1_out = self.branch1(x)
        branch2_out = self.branch2(x)
        branch3_out = self.branch3(x)

        # Concatenate the branches
        concat_input = paddle.concat([branch1_out, branch2_out, branch3_out], axis=1)
        concat_out = self.concat_conv(concat_input)

        # Direct connection
        direct_out = self.direct_connection(x)

        # Fuse the main path and the branch
        fuse_out = self.fuse_add(concat_out, direct_out)

        # Flatten for fully connected layers
        flatten_out = paddle.flatten(fuse_out, [1, 2])

        # Fully connected layers
        fc1_out = self.fc1(flatten_out)
        fc2_out = self.fc2(fc1_out)
        fc3_out = self.fc3(fc2_out)

        return fc3_out


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn(shape=[1, 3, 32, 32])
    output_tensor = model(input_tensor)
    print(output_tensor.shape)