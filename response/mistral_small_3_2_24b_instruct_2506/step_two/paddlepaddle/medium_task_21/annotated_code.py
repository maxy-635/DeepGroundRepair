import paddle
import paddle.nn as nn

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Branch 1: 1x1 convolution
        self.branch1 = nn.Sequential(
            nn.Conv2D(in_channels=3, out_channels=64, kernel_size=1, stride=1, padding=0),
            nn.Dropout(p=0.5)
        )

        # Branch 2: 1x1 convolution followed by 3x3 convolution
        self.branch2 = nn.Sequential(
            nn.Conv2D(in_channels=3, out_channels=64, kernel_size=1, stride=1, padding=0),
            nn.Conv2D(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.Dropout(p=0.5)
        )

        # Branch 3: 1x1 convolution followed by two consecutive 3x3 convolutions
        self.branch3 = nn.Sequential(
            nn.Conv2D(in_channels=3, out_channels=64, kernel_size=1, stride=1, padding=0),
            nn.Conv2D(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.Conv2D(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.Dropout(p=0.5)
        )

        # Branch 4: Average pooling followed by 1x1 convolution
        self.branch4 = nn.Sequential(
            nn.AvgPool2D(kernel_size=3, stride=1, padding=1),
            nn.Conv2D(in_channels=3, out_channels=64, kernel_size=1, stride=1, padding=0),
            nn.Dropout(p=0.5)
        )

        # Fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Linear(in_features=64 * 4 * 32 * 32, out_features=1024),
            nn.Linear(in_features=1024, out_features=512),
            nn.Linear(in_features=512, out_features=10)
        )

    def forward(self, x):
        # Process input through each branch
        branch1_out = self.branch1(x)
        branch2_out = self.branch2(x)
        branch3_out = self.branch3(x)
        branch4_out = self.branch4(x)

        # Concatenate the outputs of all branches
        concatenated = paddle.concat([branch1_out, branch2_out, branch3_out, branch4_out], axis=1)

        # Flatten the concatenated features
        flattened = paddle.flatten(concatenated, start_axis=1, stop_axis=-1)

        # Pass through fully connected layers
        output = self.fc_layers(flattened)

        return output

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn(shape=[1, 3, 32, 32])  # CIFAR-10 image shape: [batch_size, 3, 32, 32]
    output_tensor = model(input_tensor)
    print("Output shape:", output_tensor.shape)  # Expected output shape: [1, 10]