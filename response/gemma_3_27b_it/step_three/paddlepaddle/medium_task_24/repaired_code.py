import paddle
import paddle.nn as nn
import paddle.nn.functional as F

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Branch 1: 1x1 convolution, 3x3 convolution
        self.branch1_conv1 = nn.Conv2D(in_channels=3, out_channels=32, kernel_size=1, stride=1, padding=0)
        self.branch1_conv2 = nn.Conv2D(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.branch1_dropout = nn.Dropout(p=0.2)

        # Branch 2: 1x1 convolution, 1x7 convolution, 7x1 convolution, 3x3 convolution
        self.branch2_conv1 = nn.Conv2D(in_channels=3, out_channels=32, kernel_size=1, stride=1, padding=0)
        self.branch2_conv2 = nn.Conv2D(in_channels=32, out_channels=64, kernel_size=(1, 7), stride=1, padding=(0, 3))
        self.branch2_conv3 = nn.Conv2D(in_channels=64, out_channels=64, kernel_size=(7, 1), stride=1, padding=(3, 0))
        self.branch2_conv4 = nn.Conv2D(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.branch2_dropout = nn.Dropout(p=0.2)

        # Branch 3: Max pooling
        self.branch3_pool = nn.MaxPool2D(kernel_size=1, stride=1, padding=0)
        self.branch3_dropout = nn.Dropout(p=0.2)

        # Fully connected layers
        self.fc1 = nn.Linear(in_features=64 * 8 * 8, out_features=128)  # Adjust input features based on pooling
        self.fc2 = nn.Linear(in_features=128, out_features=64)
        self.fc3 = nn.Linear(in_features=64, out_features=10)

    def forward(self, x):
        # the initial shape of x is: (1, 3, 32, 32)
        # Process through branches
        branch1_output = F.relu(self.branch1_conv2(F.relu(self.branch1_conv1(x))))  # output tensor shape: (1, 64, 32, 32)
        branch1_output = self.branch1_dropout(branch1_output)  # output tensor shape: (1, 64, 32, 32)

        branch2_output = F.relu(self.branch2_conv4(F.relu(self.branch2_conv3(F.relu(self.branch2_conv2(F.relu(self.branch2_conv1(x))))))))  # output tensor shape: (1, 64, 32, 32)
        branch2_output = self.branch2_dropout(branch2_output)  # output tensor shape: (1, 64, 32, 32)

        branch3_output = self.branch3_dropout(self.branch3_pool(x))  # output tensor shape of this layer should be (1, 3, 32, 32)

        # Concatenate outputs
        concat_output = paddle.concat([branch1_output, branch2_output, branch3_output], axis=1)

        # Flatten and pass through fully connected layers
        flattened_output = paddle.reshape(concat_output, [concat_output.shape[0], -1])
        fc1_output = F.relu(self.fc1(flattened_output))
        fc2_output = F.relu(self.fc2(fc1_output))
        output = self.fc3(fc2_output)

        return output


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn(shape=[1, 3, 32, 32])  # Batch size of 1, 3 channels, 32x32 image
    output_tensor = model(input_tensor)
    print(output_tensor.shape)