import paddle
import paddle.nn as nn
import paddle.nn.functional as F

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Block 1: Three parallel paths with different average pooling scales
        self.avg_pool1x1 = nn.AvgPool2D(kernel_size=1, stride=1, padding=0)
        self.avg_pool2x2 = nn.AvgPool2D(kernel_size=2, stride=2, padding=0)
        self.avg_pool4x4 = nn.AvgPool2D(kernel_size=4, stride=4, padding=0)
        self.flatten = nn.Flatten()

        # Fully connected layer between Block 1 and Block 2
        self.fc1 = nn.Linear(in_features=784, out_features=128)

        # Reshape transform
        self.reshape = paddle.distribution.ReshapeTransform(in_event_shape=[128], out_event_shape=[1, 128, 1, 1])

        # Block 2: Three branches for feature extraction
        # Branch 1: 1x1 convolution followed by 3x3 convolution
        self.branch1_conv1x1 = nn.Conv2D(in_channels=1, out_channels=64, kernel_size=1, stride=1, padding=0)
        self.branch1_conv3x3 = nn.Conv2D(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)

        # Branch 2: 1x1 convolution followed by 1x7 convolution, 7x1 convolution, and 3x3 convolution
        self.branch2_conv1x1 = nn.Conv2D(in_channels=1, out_channels=64, kernel_size=1, stride=1, padding=0)
        self.branch2_conv1x7 = nn.Conv2D(in_channels=64, out_channels=64, kernel_size=(1, 7), stride=1, padding=(0, 3))
        self.branch2_conv7x1 = nn.Conv2D(in_channels=64, out_channels=64, kernel_size=(7, 1), stride=1, padding=(3, 0))
        self.branch2_conv3x3 = nn.Conv2D(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)

        # Branch 3: Average pooling
        self.branch3_avg_pool = nn.AvgPool2D(kernel_size=3, stride=1, padding=1)

        # Final fully connected layers
        self.fc2 = nn.Linear(in_features=192, out_features=128)
        self.fc3 = nn.Linear(in_features=128, out_features=10)

    def forward(self, x):
        # Block 1: Three parallel paths with different average pooling scales
        x1 = self.avg_pool1x1(x)
        x2 = self.avg_pool2x2(x)
        x3 = self.avg_pool4x4(x)

        # Flatten and concatenate the outputs
        x1_flat = self.flatten(x1)
        x2_flat = self.flatten(x2)
        x3_flat = self.flatten(x3)
        x_combined = paddle.concat([x1_flat, x2_flat, x3_flat], axis=1)

        # Fully connected layer between Block 1 and Block 2
        x_fc = self.fc1(x_combined)

        # Reshape the output for Block 2
        x_reshape = self.reshape(x_fc)

        # Block 2: Three branches for feature extraction
        # Branch 1: 1x1 convolution followed by 3x3 convolution
        branch1 = self.branch1_conv3x3(F.relu(self.branch1_conv1x1(x_reshape)))

        # Branch 2: 1x1 convolution followed by 1x7 convolution, 7x1 convolution, and 3x3 convolution
        branch2 = self.branch2_conv3x3(F.relu(self.branch2_conv7x1(F.relu(self.branch2_conv1x7(F.relu(self.branch2_conv1x1(x_reshape)))))))

        # Branch 3: Average pooling
        branch3 = self.branch3_avg_pool(x_reshape)

        # Concatenate the outputs of all branches
        x_branches = paddle.concat([branch1, branch2, branch3], axis=1)

        # Flatten the concatenated output
        x_flat = self.flatten(x_branches)

        # Final fully connected layers
        x_fc2 = self.fc2(x_flat)
        x_fc3 = self.fc3(x_fc2)

        return x_fc3

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn([1, 1, 28, 28])  # Dummy input tensor for MNIST (batch_size=1, channels=1, height=28, width=28)
    output_tensor = model(input_tensor)
    print(output_tensor.shape)  # Expected output shape: [1, 10]