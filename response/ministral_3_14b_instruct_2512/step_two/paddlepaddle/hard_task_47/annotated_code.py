import paddle
import paddle.nn as nn
import paddle.nn.functional as F

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()

        # First block: Three groups with different kernel sizes
        self.group1 = nn.Sequential(
            nn.Conv2D(in_channels=3, out_channels=64, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2D(num_features=64),
            nn.ReLU()
        )

        self.group2 = nn.Sequential(
            nn.Conv2D(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2D(num_features=64),
            nn.ReLU()
        )

        self.group3 = nn.Sequential(
            nn.Conv2D(in_channels=3, out_channels=64, kernel_size='mask', stride='mask', padding='mask'),
            nn.BatchNorm2D(num_features=64),
            nn.ReLU()
        )

        # Second block: Multiple branches
        self.branch1 = nn.Sequential(
            nn.Conv2D(in_channels=192, out_channels=64, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2D(num_features=64),
            nn.ReLU()
        )

        self.branch2 = nn.Sequential(
            nn.Conv2D(in_channels=192, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2D(num_features=64),
            nn.ReLU()
        )

        self.branch3 = nn.Sequential(
            nn.Conv2D(in_channels=192, out_channels=64, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2D(num_features=64),
            nn.ReLU(),
            nn.Conv2D(in_channels=64, out_channels=64, kernel_size=(1, 7), stride=1, padding=(0, 3)),
            nn.BatchNorm2D(num_features=64),
            nn.ReLU(),
            nn.Conv2D(in_channels=64, out_channels=64, kernel_size=(7, 1), stride=1, padding=(3, 0)),
            nn.BatchNorm2D(num_features=64),
            nn.ReLU(),
            nn.Conv2D(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2D(num_features=64),
            nn.ReLU()
        )

        self.branch4 = nn.Sequential(
            nn.LPPool2D(norm_type=2, kernel_size=3, stride=1)
        )

        # Fully connected layers
        self.fc1 = nn.Linear(in_features=64*32*32, out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=10)

    def forward(self, x):
        # the initial shape of x is: (1, 3, 32, 32)
        # First block: Process three groups and concatenate
        x1 = self.group1(x)  # output tensor shape: (1, 64, 32, 32)
        x2 = self.group2(x)  # output tensor shape: (1, 64, 32, 32)
        x3 = self.group3(x)  # output tensor shape of this layer should be (1, 64, 32, 32)
        x_concat = paddle.concat([x1, x2, x3], axis=1)

        # Second block: Process through multiple branches
        b1 = self.branch1(x_concat)  # output tensor shape: (1, 64, 32, 32)
        b2 = self.branch2(x_concat)  # output tensor shape: (1, 64, 32, 32)
        b3 = self.branch3(x_concat)  # output tensor shape: (1, 64, 32, 32)
        b4 = self.branch4(x_concat)

        # Concatenate all branches
        x_final = paddle.concat([b1, b2, b3, b4], axis=1)

        # Flatten and pass through fully connected layers
        x_flat = paddle.flatten(x_final, start_axis=1)
        x_fc1 = self.fc1(x_flat)
        x_fc1 = F.relu(x_fc1)
        x_fc2 = self.fc2(x_fc1)

        return x_fc2

if __name__ == "__main__":
    model = DL_Model()
    # Create a dummy input tensor with shape [batch_size, 3, 32, 32] for CIFAR-10
    input_tensor = paddle.randn((1, 3, 32, 32))
    output_tensor = model(input_tensor)
    print("Input shape:", input_tensor.shape)
    print("Output shape:", output_tensor.shape)