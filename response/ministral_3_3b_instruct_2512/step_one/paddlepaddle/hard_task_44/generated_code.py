import paddle
import paddle.nn as nn

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()
        # Block 1: Splitting and feature extraction
        self.conv1_1x1 = nn.Conv2D(in_channels=3, out_channels=64, kernel_size=1, stride=1, padding=0)
        self.conv1_3x3 = nn.Conv2D(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv1_5x5 = nn.Conv2D(in_channels=3, out_channels=64, kernel_size=5, stride=1, padding=2)
        self.dropout1 = nn.Dropout(p=0.2)

        # Block 2: Four branches for feature fusion
        self.branch1 = nn.Sequential(
            nn.Conv2D(in_channels=3, out_channels=64, kernel_size=1, stride=1, padding=0),
            nn.ReLU()
        )

        self.branch2 = nn.Sequential(
            nn.Conv2D(in_channels=3, out_channels=64, kernel_size=1, stride=1, padding=0),
            nn.Conv2D(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )

        self.branch3 = nn.Sequential(
            nn.Conv2D(in_channels=3, out_channels=64, kernel_size=1, stride=1, padding=0),
            nn.Conv2D(in_channels=64, out_channels=64, kernel_size=5, stride=1, padding=2),
            nn.ReLU()
        )

        self.branch4 = nn.Sequential(
            nn.MaxPool2D(kernel_size=3, stride=3, padding=1),
            nn.Conv2D(in_channels=3, out_channels=64, kernel_size=1, stride=1, padding=0),
            nn.ReLU()
        )

        # Concatenation layers for combining outputs
        self.concat1 = nn.Sequential(
            nn.Conv2D(in_channels=192, out_channels=128, kernel_size=1, stride=1, padding=0),
            nn.ReLU()
        )

        # Final layers
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(in_features=128 * 32 * 32, out_features=10)

    def forward(self, x):
        # Block 1: Splitting and feature extraction
        conv1_1x1 = self.conv1_1x1(x)
        conv1_3x3 = self.conv1_3x3(x)
        conv1_5x5 = self.conv1_5x5(x)

        # Concatenate outputs from three branches
        concat1 = paddle.concat([conv1_1x1, conv1_3x3, conv1_5x5], axis=1)
        concat1 = self.dropout1(concat1)
        concat1 = self.concat1(concat1)

        # Block 2: Four branches for feature fusion
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)

        # Concatenate outputs from four branches
        branch_outputs = paddle.concat([branch1, branch2, branch3, branch4], axis=1)
        branch_outputs = self.concat1(branch_outputs)

        # Combine outputs from both blocks
        combined_output = paddle.concat([concat1, branch_outputs], axis=1)

        # Final processing
        output = self.flatten(combined_output)
        output = self.fc(output)

        return output

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn((1, 3, 32, 32))  # Example input tensor with shape (batch_size, channels, height, width)
    output_tensor = model(input_tensor)
    print("Input shape:", input_tensor.shape)
    print("Output shape:", output_tensor.shape)