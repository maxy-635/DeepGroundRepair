import paddle
import paddle.nn as nn
import paddle.nn.functional as F

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Block 1: Three parallel average pooling paths
        self.avg_pool1x1 = nn.AvgPool2D(kernel_size=1, stride=1, padding=0)
        self.avg_pool2x2 = nn.AvgPool2D(kernel_size=2, stride=2, padding=0)
        self.avg_pool4x4 = nn.AvgPool2D(kernel_size=4, stride=4, padding=0)

        # Flatten and concatenate
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_features=1*1*28*28 + 1*1*14*14 + 1*1*7*7, out_features=128)

        # Reshape for Block 2
        self.reshape = paddle.reshape_

        # Block 2: Three branches for feature extraction
        # Branch 1: 1x1 conv + 3x3 conv
        self.branch1_conv1x1 = nn.Conv2D(in_channels=128, out_channels=64, kernel_size=1, stride=1, padding=0)
        self.branch1_conv3x3 = nn.Conv2D(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)

        # Branch 2: 1x1 conv + 1x7 conv + 7x1 conv + 3x3 conv
        self.branch2_conv1x1 = nn.Conv2D(in_channels=128, out_channels=64, kernel_size=1, stride=1, padding=0)
        self.branch2_conv1x7 = nn.Conv2D(in_channels=64, out_channels=64, kernel_size=(1, 7), stride=1, padding=(0, 3))
        self.branch2_conv7x1 = nn.Conv2D(in_channels=64, out_channels=64, kernel_size=(7, 1), stride=1, padding=(3, 0))
        self.branch2_conv3x3 = nn.Conv2D(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)

        # Branch 3: Average pooling
        self.branch3_avg_pool = nn.AvgPool2D(kernel_size=3, stride=1, padding=1)

        # Final classification layers
        self.fc2 = nn.Linear(in_features=64*28*28*3, out_features=10)  # 3 branches, each with 64 channels

    def forward(self, x):
        # Block 1 processing
        pool1 = self.avg_pool1x1(x)
        pool2 = self.avg_pool2x2(x)
        pool4 = self.avg_pool4x4(x)

        # Flatten and concatenate
        pool1_flat = self.flatten(pool1)
        pool2_flat = self.flatten(pool2)
        pool4_flat = self.flatten(pool4)

        # Concatenate along channel dimension
        concat = paddle.concat([pool1_flat, pool2_flat, pool4_flat], axis=1)

        # Fully connected layer
        fc_out = self.fc1(concat)

        # Reshape for Block 2
        fc_out = self.reshape(fc_out, shape=[-1, 128, 1, 1])
        fc_out = paddle.nn.functional.interpolate(fc_out, size=(28, 28), mode='bilinear', align_corners=False)

        # Block 2 processing
        # Branch 1
        branch1 = self.branch1_conv1x1(fc_out)
        branch1 = self.branch1_conv3x3(branch1)

        # Branch 2
        branch2 = self.branch2_conv1x1(fc_out)
        branch2 = self.branch2_conv1x7(branch2)
        branch2 = self.branch2_conv7x1(branch2)
        branch2 = self.branch2_conv3x3(branch2)

        # Branch 3
        branch3 = self.branch3_avg_pool(fc_out)

        # Concatenate all branches
        concat = paddle.concat([branch1, branch2, branch3], axis=1)

        # Flatten and final classification
        concat_flat = self.flatten(concat)
        logits = self.fc2(concat_flat)

        return logits

if __name__ == "__main__":
    model = DL_Model()
    # Create a dummy input tensor with shape [batch_size, channels, height, width]
    # MNIST images are 28x28 with 1 channel
    input_tensor = paddle.randn(shape=[1, 1, 28, 28])
    output_tensor = model(input_tensor)
    print("Output shape:", output_tensor.shape)