import paddle
import paddle.nn as nn
import paddle.nn.functional as F

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Path1: Two blocks of convolution followed by average pooling
        self.path1_conv1 = nn.Conv2D(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.path1_conv2 = nn.Conv2D(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.path1_pool = nn.AvgPool2D(kernel_size=2, stride=2, padding=0)

        # Path2: Single convolutional layer
        self.path2_conv = nn.Conv2D(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)

        # Flatten layers
        self.flatten = nn.Flatten()

        # Fully connected layer
        self.fc = nn.Linear(in_features=64*8*8 + 32*8*8, out_features=10)

    def forward(self, x):
        # Path1 processing
        path1_out = self.path1_conv1(x)
        path1_out = F.relu(path1_out)
        path1_out = self.path1_conv2(path1_out)
        path1_out = F.relu(path1_out)
        path1_out = self.path1_pool(path1_out)

        # Path2 processing
        path2_out = self.path2_conv(x)
        path2_out = F.relu(path2_out)
        path2_out = self.path1_pool(path2_out)

        # Combine the outputs
        combined = paddle.concat([path1_out, path2_out], axis=1)

        # Flatten and pass through fully connected layer
        flattened = self.flatten(combined)
        output = self.fc(flattened)

        return output

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn((1, 3, 32, 32))  # Batch size 1, 3 channels, 32x32 image
    output_tensor = model(input_tensor)
    print(output_tensor.shape)