import paddle
import paddle.nn as nn
import paddle.nn.functional as F

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()

        # First Block: Depthwise Separable Convolutions
        self.group1 = nn.Sequential(
            nn.Conv2D(in_channels=3, out_channels=32, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2D(num_features=32),
            F.leaky_relu(negative_slope=0.2)
        )
        self.group2 = nn.Sequential(
            nn.Conv2D(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2D(num_features=32),
            F.leaky_relu(negative_slope=0.2)
        )
        self.group3 = nn.Sequential(
            nn.Conv2D(in_channels=3, out_channels=32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2D(num_features=32),
            F.leaky_relu(negative_slope=0.2)
        )

        # Concatenate the outputs of the three groups
        # Second Block: Multiple Branches
        self.branch1 = nn.Sequential(
            nn.Conv2D(in_channels=96, out_channels=64, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2D(num_features=64),
            F.leaky_relu(negative_slope=0.2)
        )
        self.branch2 = nn.Sequential(
            nn.Conv2D(in_channels=96, out_channels=64, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2D(num_features=64),
            F.leaky_relu(negative_slope=0.2),
            nn.Conv2D(in_channels=64, out_channels=64, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm2D(num_features=64),
            F.leaky_relu(negative_slope=0.2),
            nn.Conv2D(in_channels=64, out_channels=64, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2D(num_features=64),
            F.leaky_relu(negative_slope=0.2),
            nn.Conv2D(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2D(num_features=64),
            F.leaky_relu(negative_slope=0.2)
        )
        self.branch3 = nn.Sequential(
            nn.AdaptiveAvgPool2D(output_size=(1, 1))
        )

        # Fully Connected Layers
        self.fc1 = nn.Linear(in_features=192, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=10)

    def forward(self, x):
        # First Block
        out1 = self.group1(x)
        out2 = self.group2(x)
        out3 = self.group3(x)
        merged_output = paddle.concat([out1, out2, out3], axis=-1)

        # Second Block
        branch1_output = self.branch1(merged_output)
        branch2_output = self.branch2(merged_output)
        branch3_output = self.branch3(merged_output)
        
        # Concatenate the outputs from all branches
        final_output = paddle.concat([branch1_output, branch2_output, branch3_output], axis=-1)

        # Fully Connected Layers
        x = F.leaky_relu(negative_slope=0.2)(self.fc1(final_output))
        x = self.fc2(x)
        return x


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.rand(shape=(1, 3, 32, 32))  # Dummy input tensor
    output_tensor = model(input_tensor)
    print(output_tensor.shape)