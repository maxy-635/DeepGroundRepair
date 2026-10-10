import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle import concat

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Block 1: Max Pooling Layers
        self.max_pool1 = nn.MaxPool2D(kernel_size=1, stride=1, padding=0)
        self.max_pool2 = nn.MaxPool2D(kernel_size=2, stride=2, padding=0)
        self.max_pool3 = nn.MaxPool2D(kernel_size=4, stride=4, padding=0)
        self.flatten1 = nn.Flatten()
        self.flatten2 = nn.Flatten()
        self.flatten3 = nn.Flatten()

        # Fully Connected Layer and Reshape
        self.fc1 = nn.Linear(in_features=1029, out_features=128)
        self.reshape = paddle.distribution.ReshapeTransform(in_event_shape=[128], out_event_shape=[128, 1, 1, 1])

        # Block 2: Multiple Branches
        self.conv1x1 = nn.Conv2D(in_channels=128, out_channels=32, kernel_size=1, stride=1, padding=0)
        self.conv3x3 = nn.Conv2D(in_channels=128, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv5x5 = nn.Conv2D(in_channels=128, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.max_pool_3x3 = nn.MaxPool2D(kernel_size=3, stride=3, padding=1)

        # Flatten and Fully Connected Layer for Classification
        self.flatten4 = nn.Flatten()
        self.fc2 = nn.Linear(in_features=32 * 32 * 32, out_features=10)  # Assuming output size from Block 2 is 32x32

    def forward(self, x):
        # the initial shape of x is: (1, 1, 28, 28)
        # Block 1
        x1 = self.max_pool1(x)  # output tensor shape: (1, 1, 28, 28)
        x2 = self.max_pool2(x)  # output tensor shape: (1, 1, 14, 14)
        x3 = self.max_pool3(x)  # output tensor shape: (1, 1, 7, 7)

        x1_flat = self.flatten1(x1)  # output tensor shape: (1, 784)
        x2_flat = self.flatten2(x2)  # output tensor shape: (1, 196)
        x3_flat = self.flatten3(x3)  # output tensor shape: (1, 49)
        
        block1_output = concat([x1_flat, x2_flat, x3_flat], axis=1)  # output tensor shape: (1, 1029)

        # Fully Connected Layer and Reshape
        block1_output = self.fc1(block1_output)
        block1_output = self.reshape(block1_output)

        # Block 2
        branch1 = self.conv1x1(block1_output)
        branch2 = self.conv3x3(block1_output)
        branch3 = self.conv5x5(block1_output)
        branch4 = self.max_pool_3x3(block1_output)

        block2_output = concat([branch1, branch2, branch3, branch4], axis=1)

        # Flatten and Fully Connected Layer for Classification
        block2_output = self.flatten4(block2_output)
        output = self.fc2(block2_output)

        return output


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn([1, 1, 28, 28])  # Batch size 1, 1 channel, 28x28 image
    output_tensor = model(input_tensor)
    print(output_tensor.shape)