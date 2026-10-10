import paddle
import paddle.nn as nn
import paddle.nn.functional as F

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Block 1: Three parallel average pooling paths
        self.avg_pool_1x1 = nn.AvgPool2D(kernel_size=1, stride=1, padding=0)
        self.avg_pool_2x2 = nn.AvgPool2D(kernel_size=2, stride=2, padding=0)
        self.avg_pool_4x4 = nn.AvgPool2D(kernel_size=4, stride=4, padding=0)

        self.dropout = nn.Dropout(p=0.5)

        # Fully connected layer and reshape for block 2 input
        self.fc = nn.Linear(in_features=1029, out_features=100)
        self.reshape = paddle.distribution.ReshapeTransform(in_event_shape=[1, 100], out_event_shape=[1, 1, 10, 10])

        # Block 2: Four branches
        # Branch 1: 1x1 convolution
        self.branch1_conv1x1 = nn.Conv2D(in_channels=1, out_channels=64, kernel_size=1, stride=1, padding=0)

        # Branch 2: 1x1 convolution followed by 3x3 convolution
        self.branch2_conv1x1 = nn.Conv2D(in_channels=1, out_channels=64, kernel_size=1, stride=1, padding=0)
        self.branch2_conv3x3 = nn.Conv2D(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)

        # Branch 3: 1x1 convolution followed by two 3x3 convolutions
        self.branch3_conv1x1 = nn.Conv2D(in_channels=1, out_channels=64, kernel_size=1, stride=1, padding=0)
        self.branch3_conv3x3_1 = nn.Conv2D(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.branch3_conv3x3_2 = nn.Conv2D(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)

        # Branch 4: Average pooling followed by 1x1 convolution
        self.branch4_avg_pool = nn.AvgPool2D(kernel_size=2, stride=2, padding=0)
        self.branch4_conv1x1 = nn.Conv2D(in_channels=1, out_channels=64, kernel_size=1, stride=1, padding=0)

        # Final fully connected layers
        self.fc_final1 = nn.Linear(in_features=256, out_features=128)  # Assuming concatenated features are 256
        self.fc_final2 = nn.Linear(in_features=128, out_features=10)   # Output layer for 10 classes

    def forward(self, x):
        # the initial shape of x is: (1, 1, 28, 28)
        # Block 1 processing
        pool1 = self.avg_pool_1x1(x)  # output tensor shape: (1, 1, 28, 28)
        pool2 = self.avg_pool_2x2(x)  # output tensor shape: (1, 1, 14, 14)
        pool3 = self.avg_pool_4x4(x)  # output tensor shape: (1, 1, 7, 7)

        # Flatten and apply dropout
        pool1_flat = paddle.flatten(pool1, start_axis=1, stop_axis=-1)  # output tensor shape: (1, 784)
        pool2_flat = paddle.flatten(pool2, start_axis=1, stop_axis=-1)  # output tensor shape: (1, 196)
        pool3_flat = paddle.flatten(pool3, start_axis=1, stop_axis=-1)  # output tensor shape: (1, 49)

        pool1_drop = self.dropout(pool1_flat)  # output tensor shape: (1, 784)
        pool2_drop = self.dropout(pool2_flat)  # output tensor shape: (1, 196)
        pool3_drop = self.dropout(pool3_flat)  # output tensor shape: (1, 49)

        # Concatenate the outputs
        block1_output = paddle.concat([pool1_drop, pool2_drop, pool3_drop], axis=1)  # output tensor shape: (1, 1029)

        # Fully connected layer and reshape
        fc_output = self.fc(block1_output)
        reshaped = self.reshape(fc_output)

        # Block 2 processing
        # Branch 1
        branch1 = self.branch1_conv1x1(reshaped)

        # Branch 2
        branch2_conv1x1 = self.branch2_conv1x1(reshaped)
        branch2_conv3x3 = self.branch2_conv3x3_1(branch2_conv1x1)

        # Branch 3
        branch3_conv1x1 = self.branch3_conv1x1(reshaped)
        branch3_conv3x3_1 = self.branch3_conv3x3_1(branch3_conv1x1)
        branch3_conv3x3_2 = self.branch3_conv3x3_2(branch3_conv3x3_1)

        # Branch 4
        branch4_avg_pool = self.branch4_avg_pool(reshaped)
        branch4_conv1x1 = self.branch4_conv1x1(branch4_avg_pool)

        # Concatenate all branches
        block2_output = paddle.concat([branch1, branch2_conv3x3, branch3_conv3x3_2, branch4_conv1x1], axis=1)

        # Flatten and final fully connected layers
        block2_flat = paddle.flatten(block2_output, start_axis=1, stop_axis=-1)
        fc_final1 = self.fc_final1(block2_flat)
        fc_final2 = self.fc_final2(fc_final1)

        return fc_final2

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn(shape=[1, 1, 28, 28])  # MNIST image shape: batch_size=1, channels=1, height=28, width=28
    output_tensor = model(input_tensor)
    print("Output shape:", output_tensor.shape)