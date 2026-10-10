import paddle
import paddle.nn as nn

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Define the three groups of convolutions
        self.conv1x1_group1 = nn.Conv2D(in_channels=1, out_channels=64, kernel_size=1, stride=1, padding=0)
        self.conv3x3_group1 = nn.Conv2D(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv1x1_group1_final = nn.Conv2D(in_channels=64, out_channels=64, kernel_size=1, stride=1, padding=0)

        self.conv1x1_group2 = nn.Conv2D(in_channels=3, out_channels=64, kernel_size=1, stride=1, padding=0)
        self.conv3x3_group2 = nn.Conv2D(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv1x1_group2_final = nn.Conv2D(in_channels=64, out_channels=64, kernel_size=1, stride=1, padding=0)

        self.conv1x1_group3 = nn.Conv2D(in_channels=3, out_channels=64, kernel_size=1, stride=1, padding=0)
        self.conv3x3_group3 = nn.Conv2D(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv1x1_group3_final = nn.Conv2D(in_channels=64, out_channels=64, kernel_size=1, stride=1, padding=0)

        # Define the fully connected layer for classification
        self.fc = nn.Linear(in_features=64 * 32 * 32, out_features=10)

    def forward(self, x):
        # the initial shape of x is: (1, 3, 32, 32)
        # Split the input into three groups along the channel dimension
        x1 = x[:, :1, :, :]  # output tensor shape: (1, 1, 32, 32)
        x2 = x[:, 1:2, :, :]  # output tensor shape: (1, 1, 32, 32)
        x3 = x[:, 2:, :, :]  # output tensor shape: (1, 1, 32, 32)

        # Process each group through the convolutional layers
        out1 = self.conv1x1_group1(x1)
        out1 = self.conv3x3_group1(out1)
        out1 = self.conv1x1_group1_final(out1)

        out2 = self.conv1x1_group2(x2)
        out2 = self.conv3x3_group2(out2)
        out2 = self.conv1x1_group2_final(out2)

        out3 = self.conv1x1_group3(x3)
        out3 = self.conv3x3_group3(out3)
        out3 = self.conv1x1_group3_final(out3)

        # Combine the outputs from the three groups using addition
        combined = paddle.add(out1, out2)
        combined = paddle.add(combined, out3)

        # Fuse with the original input
        fused = paddle.add(combined, x)

        # Flatten the combined features
        flattened = paddle.flatten(fused, start_axis=1, stop_axis=-1)

        # Pass through the fully connected layer for classification
        output = self.fc(flattened)

        return output

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn((1, 3, 32, 32))  # Batch size of 1, 3 channels, 32x32 image
    output_tensor = model(input_tensor)
    print(output_tensor.shape)  # Expected output shape: [1, 10]