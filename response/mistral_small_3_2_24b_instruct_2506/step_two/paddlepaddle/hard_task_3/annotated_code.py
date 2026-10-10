import paddle
import paddle.nn as nn
import paddle.nn.functional as F

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Main pathway - three groups
        self.group1_conv1 = nn.Conv2D(in_channels=3, out_channels=64, kernel_size=1, stride=1, padding=0)
        self.group1_conv2 = nn.Conv2D(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.group1_dropout = nn.Dropout(p='mask')

        self.group2_conv1 = nn.Conv2D(in_channels=3, out_channels=64, kernel_size=1, stride=1, padding=0)
        self.group2_conv2 = nn.Conv2D(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.group2_dropout = nn.Dropout(p=0.5)

        self.group3_conv1 = nn.Conv2D(in_channels=3, out_channels=64, kernel_size=1, stride=1, padding=0)
        self.group3_conv2 = nn.Conv2D(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.group3_dropout = nn.Dropout(p=0.5)

        # Concatenation layer
        self.concat = nn.Sequential()

        # Branch pathway
        self.branch_conv = nn.Conv2D(in_channels=3, out_channels=192, kernel_size=1, stride=1, padding=0)

        # Final fully connected layer
        self.fc = nn.Linear(in_features=192 * 32 * 32, out_features=10)

    def forward(self, x):
        # the initial shape of x is: (1, 3, 32, 32)
        # Split input into three groups
        x1 = x[:, :1, :, :]  # output tensor shape: (1, 1, 32, 32)
        x2 = x[:, 1:2, :, :]  # output tensor shape: (1, 1, 32, 32)
        x3 = x[:, 2:, :, :]  # output tensor shape: (1, 1, 32, 32)

        # Process each group
        out1 = self.group1_dropout(F.relu(self.group1_conv2(F.relu(self.group1_conv1(x1)))))
        out2 = self.group2_dropout(F.relu(self.group2_conv2(F.relu(self.group2_conv1(x2)))))
        out3 = self.group3_dropout(F.relu(self.group3_conv2(F.relu(self.group3_conv1(x3)))))

        # Concatenate outputs
        out = paddle.concat([out1, out2, out3], axis=1)

        # Branch pathway
        branch_out = self.branch_conv(x)

        # Combine pathways
        combined = paddle.add(out, branch_out)

        # Flatten and pass through FC layer
        combined = paddle.flatten(combined, start_axis=1)
        output = self.fc(combined)

        return output

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn([1, 3, 32, 32])  # CIFAR-10 input shape
    output_tensor = model(input_tensor)
    print(output_tensor.shape)  # Expected output shape: [1, 10]