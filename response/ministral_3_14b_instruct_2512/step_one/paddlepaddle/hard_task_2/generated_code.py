import paddle
import paddle.nn as nn
import paddle.nn.functional as F

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Split the input into three groups along channel dimension
        self.split_conv1 = nn.Conv2D(in_channels=3, out_channels=3, kernel_size=1, stride=1, padding=0)
        self.split_conv2 = nn.Conv2D(in_channels=3, out_channels=3, kernel_size=1, stride=1, padding=0)
        self.split_conv3 = nn.Conv2D(in_channels=3, out_channels=3, kernel_size=1, stride=1, padding=0)

        # First 1x1 conv for each group
        self.conv1_1 = nn.Conv2D(in_channels=3, out_channels=64, kernel_size=1, stride=1, padding=0)
        self.conv1_2 = nn.Conv2D(in_channels=3, out_channels=64, kernel_size=1, stride=1, padding=0)
        self.conv1_3 = nn.Conv2D(in_channels=3, out_channels=64, kernel_size=1, stride=1, padding=0)

        # 3x3 conv for each group
        self.conv2_1 = nn.Conv2D(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv2_2 = nn.Conv2D(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv2_3 = nn.Conv2D(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)

        # Second 1x1 conv for each group
        self.conv3_1 = nn.Conv2D(in_channels=64, out_channels=3, kernel_size=1, stride=1, padding=0)
        self.conv3_2 = nn.Conv2D(in_channels=64, out_channels=3, kernel_size=1, stride=1, padding=0)
        self.conv3_3 = nn.Conv2D(in_channels=64, out_channels=3, kernel_size=1, stride=1, padding=0)

        # BatchNorm and ReLU for each group
        self.bn1_1 = nn.BatchNorm2D(num_features=64)
        self.bn1_2 = nn.BatchNorm2D(num_features=64)
        self.bn1_3 = nn.BatchNorm2D(num_features=64)
        self.bn2_1 = nn.BatchNorm2D(num_features=64)
        self.bn2_2 = nn.BatchNorm2D(num_features=64)
        self.bn2_3 = nn.BatchNorm2D(num_features=64)
        self.bn3_1 = nn.BatchNorm2D(num_features=3)
        self.bn3_2 = nn.BatchNorm2D(num_features=3)
        self.bn3_3 = nn.BatchNorm2D(num_features=3)

        self.relu = nn.ReLU()

        # Final addition and classification layers
        self.final_conv = nn.Conv2D(in_channels=3, out_channels=3, kernel_size=1, stride=1, padding=0)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(in_features=3*32*32, out_features=10)  # Assuming CIFAR-10 input size is 32x32

    def forward(self, x):
        # Split the input into three groups along channel dimension
        x1 = self.split_conv1(x[:, :1, :, :])
        x2 = self.split_conv2(x[:, 1:2, :, :])
        x3 = self.split_conv3(x[:, 2:, :, :])

        # Process each group
        # Group 1
        x1 = self.conv1_1(x1)
        x1 = self.bn1_1(x1)
        x1 = self.relu(x1)
        x1 = self.conv2_1(x1)
        x1 = self.bn2_1(x1)
        x1 = self.relu(x1)
        x1 = self.conv3_1(x1)

        # Group 2
        x2 = self.conv1_2(x2)
        x2 = self.bn1_2(x2)
        x2 = self.relu(x2)
        x2 = self.conv2_2(x2)
        x2 = self.bn2_2(x2)
        x2 = self.relu(x2)
        x2 = self.conv3_2(x2)

        # Group 3
        x3 = self.conv1_3(x3)
        x3 = self.bn1_3(x3)
        x3 = self.relu(x3)
        x3 = self.conv2_3(x3)
        x3 = self.bn2_3(x3)
        x3 = self.relu(x3)
        x3 = self.conv3_3(x3)

        # Combine the three groups
        combined = x1 + x2 + x3

        # Add to original input
        combined = combined + x

        # Final convolution
        combined = self.final_conv(combined)
        combined = self.bn3_1(combined)
        combined = self.relu(combined)

        # Flatten and classify
        combined = self.flatten(combined)
        output = self.fc(combined)

        return output

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn((1, 3, 32, 32))  # Batch size 1, 3 channels, 32x32 image
    output_tensor = model(input_tensor)
    print("Input shape:", input_tensor.shape)
    print("Output shape:", output_tensor.shape)