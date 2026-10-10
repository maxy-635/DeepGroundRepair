import paddle
import paddle.nn as nn
import paddle.nn.functional as F

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()
        # Branch 1: 1x1 convolution for dimensionality reduction
        self.branch1 = nn.Conv2D(in_channels=3, out_channels=16, kernel_size=1, stride=1, padding=0)
        self.relu1 = nn.ReLU()

        # Branch 2: 1x1 conv + 3x3 conv
        self.branch2_conv1 = nn.Conv2D(in_channels=3, out_channels=16, kernel_size=1, stride=1, padding=0)
        self.relu2_1 = nn.ReLU()
        self.branch2_conv2 = nn.Conv2D(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.relu2_2 = nn.ReLU()

        # Branch 3: 1x1 conv + 5x5 conv
        self.branch3_conv1 = nn.Conv2D(in_channels=3, out_channels=16, kernel_size=1, stride=1, padding=0)
        self.relu3_1 = nn.ReLU()
        self.branch3_conv2 = nn.Conv2D(in_channels=16, out_channels=64, kernel_size=5, stride=1, padding=2)
        self.relu3_2 = nn.ReLU()

        # Branch 4: 3x3 max pooling + 1x1 conv
        self.branch4_pool = nn.MaxPool2D(kernel_size=3, stride=1, padding=1)
        self.branch4_conv = nn.Conv2D(in_channels=3, out_channels=32, kernel_size=1, stride=1, padding=0)
        self.relu4 = nn.ReLU()

        # Flatten layer
        self.flatten = nn.Flatten()

        # Fully connected layers
        self.fc1 = nn.Linear(in_features=16*32*32 + 32*32*32 + 64*32*32 + 32*32*32, out_features=128)
        self.relu_fc1 = nn.ReLU()
        self.fc2 = nn.Linear(in_features=128, out_features=10)

    def forward(self, x):
        # Pass input through each branch
        branch1_out = self.relu1(self.branch1(x))
        branch2_out = self.relu2_2(self.branch2_conv2(self.relu2_1(self.branch2_conv1(x))))
        branch3_out = self.relu3_2(self.branch3_conv2(self.relu3_1(self.branch3_conv1(x))))
        branch4_out = self.relu4(self.branch4_conv(self.branch4_pool(x)))

        # Concatenate the outputs
        concat_out = paddle.concat([branch1_out, branch2_out, branch3_out, branch4_out], axis=1)

        # Flatten the concatenated output
        flatten_out = self.flatten(concat_out)

        # Pass through fully connected layers
        fc1_out = self.relu_fc1(self.fc1(flatten_out))
        output = self.fc2(fc1_out)

        return output


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn(shape=[1, 3, 32, 32])  # Dummy input tensor
    output_tensor = model(input_tensor)
    print(output_tensor.shape)  # Expected output shape: [1, 10]