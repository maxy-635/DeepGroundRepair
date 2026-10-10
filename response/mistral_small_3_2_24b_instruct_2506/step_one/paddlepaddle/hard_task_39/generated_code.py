import paddle
import paddle.nn as nn

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Block 1: Three max pooling layers with different scales
        self.max_pool1 = nn.MaxPool2D(kernel_size=1, stride=1, padding=0)
        self.max_pool2 = nn.MaxPool2D(kernel_size=2, stride=2, padding=0)
        self.max_pool3 = nn.MaxPool2D(kernel_size=4, stride=4, padding=0)

        # Fully connected layer and reshape operation
        self.fc1 = nn.Linear(in_features=784, out_features=128)
        self.reshape = paddle.reshape

        # Block 2: Multiple branches with different convolutions and pooling
        self.conv1x1 = nn.Conv2D(in_channels=1, out_channels=64, kernel_size=1, stride=1, padding=0)
        self.conv3x3 = nn.Conv2D(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv5x5 = nn.Conv2D(in_channels=1, out_channels=64, kernel_size=5, stride=1, padding=2)
        self.max_pool_branch = nn.MaxPool2D(kernel_size=3, stride=1, padding=1)

        # Final classification layers
        self.flatten = nn.Flatten()
        self.fc2 = nn.Linear(in_features=256, out_features=10)

    def forward(self, x):
        # Block 1
        x1 = self.max_pool1(x)
        x2 = self.max_pool2(x)
        x3 = self.max_pool3(x)

        # Flatten and concatenate
        x1_flat = paddle.flatten(x1)
        x2_flat = paddle.flatten(x2)
        x3_flat = paddle.flatten(x3)
        x_concat = paddle.concat([x1_flat, x2_flat, x3_flat], axis=1)

        # Fully connected layer and reshape
        x_fc = self.fc1(x_concat)
        x_reshape = self.reshape(x_fc, shape=[-1, 1, 8, 8])  # Reshape to 4D tensor

        # Block 2
        branch1 = self.conv1x1(x_reshape)
        branch2 = self.conv3x3(x_reshape)
        branch3 = self.conv5x5(x_reshape)
        branch4 = self.max_pool_branch(x_reshape)

        # Concatenate branches
        x_concat2 = paddle.concat([branch1, branch2, branch3, branch4], axis=1)

        # Final classification
        x_flat = self.flatten(x_concat2)
        output = self.fc2(x_flat)

        return output

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn(shape=[1, 1, 28, 28])  # Dummy input tensor for MNIST (1 channel, 28x28)
    output_tensor = model(input_tensor)
    print(output_tensor.shape)  # Expected output shape: [1, 10]