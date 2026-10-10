import paddle
import paddle.nn as nn
import paddle.nn.functional as F

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Main path
        self.main_conv1 = nn.Conv2D(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.main_conv2 = nn.Conv2D(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.main_pool = nn.MaxPool2D(kernel_size=2, stride=2, padding=0)

        # Branch path
        self.branch_conv = nn.Conv2D(in_channels=3, out_channels=32, kernel_size='mask', stride='mask', padding='mask')

        # Fully connected layers
        self.fc1 = nn.Linear(in_features=32 * 8 * 8 * 2, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=10)

    def forward(self, x):
        # the initial shape of x is: (1, 3, 32, 32)
        # Main path
        main_out = F.relu(self.main_conv1(x))  # output tensor shape: (1, 32, 32, 32)
        main_out = F.relu(self.main_conv2(main_out))  # output tensor shape: (1, 32, 32, 32)
        main_out = self.main_pool(main_out)  # output tensor shape: (1, 32, 16, 16)

        # Branch path
        branch_out = F.relu(self.branch_conv(x))  # output tensor shape of this layer should be (1, 32, 16, 16)

        # Combine features
        combined = paddle.concat([main_out, branch_out], axis=1)

        # Flatten and fully connected layers
        combined = paddle.flatten(combined, start_axis=1)
        combined = F.relu(self.fc1(combined))
        output = self.fc2(combined)

        return output

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn([1, 3, 32, 32])  # Batch size of 1, 3 channels, 32x32 image
    output_tensor = model(input_tensor)
    print(output_tensor.shape)  # Expected output shape: [1, 10]