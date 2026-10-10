import paddle.nn as nn
import paddle

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()
        # Branch 1: 1x1 initial convolutional layer
        self.initial_conv = nn.Conv2D(in_channels=3, out_channels=64, kernel_size=1, stride=1, padding=0)

        # Branch 1: 3x3 convolutional layer
        self.branch1_conv = nn.Conv2D(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)

        # Branch 2 and 3: Downsampling and upsampling branches
        self.branch2_pool = nn.MaxPool2D(kernel_size=2, stride=2, padding=0)
        self.branch2_conv = nn.Conv2D(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.branch2_upsample = nn.Upsample(size=None, scale_factor=2, mode='bilinear', align_corners=False)

        self.branch3_pool = nn.MaxPool2D(kernel_size=2, stride=2, padding=0)
        self.branch3_conv = nn.Conv2D(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.branch3_upsample = nn.Upsample(size=None, scale_factor=2, mode='bilinear', align_corners=False)

        # Concatenation and final 1x1 convolution
        self.concatenate_conv = nn.Conv2D(in_channels=192, out_channels=64, kernel_size=1, stride=1, padding=0)

        # Fully connected layers
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_features=64 * 32 * 32, out_features=512)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.1)
        self.fc2 = nn.Linear(in_features=512, out_features=256)
        self.fc3 = nn.Linear(in_features=256, out_features=10)

    def forward(self, x):
        # Branch 1
        branch1 = self.initial_conv(x)
        branch1 = self.branch1_conv(branch1)

        # Branch 2
        branch2 = self.initial_conv(x)
        branch2 = self.branch2_pool(branch2)
        branch2 = self.branch2_conv(branch2)
        branch2 = self.branch2_upsample(branch2)

        # Branch 3
        branch3 = self.initial_conv(x)
        branch3 = self.branch3_pool(branch3)
        branch3 = self.branch3_conv(branch3)
        branch3 = self.branch3_upsample(branch3)

        # Concatenate all branches
        concatenated = nn.functional.concat([branch1, branch2, branch3], dim=1)
        concatenated = self.concatenate_conv(concatenated)

        # Fully connected layers
        x = self.flatten(concatenated)
        x = self.fc1(x)
        x = self.leaky_relu(x)
        x = self.fc2(x)
        x = self.leaky_relu(x)
        x = self.fc3(x)

        return x

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn((1, 3, 32, 32))  # Example input tensor with shape (batch_size, channels, height, width)
    output_tensor = model(input_tensor)
    print(output_tensor.shape)