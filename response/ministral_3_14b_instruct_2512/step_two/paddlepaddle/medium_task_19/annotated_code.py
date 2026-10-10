import paddle
import paddle.nn as nn
import paddle.nn.functional as F

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Branch 1: 1x1 convolution for dimensionality reduction
        self.branch1 = nn.Sequential(
            nn.Conv2D(in_channels=3, out_channels=64, kernel_size=1, stride=1, padding=0),
            nn.ReLU()
        )

        # Branch 2: 1x1 convolution followed by 3x3 convolution
        self.branch2 = nn.Sequential(
            nn.Conv2D(in_channels=3, out_channels=64, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2D(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )

        # Branch 3: 1x1 convolution followed by 5x5 convolution
        self.branch3 = nn.Sequential(
            nn.Conv2D(in_channels=3, out_channels=64, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2D(in_channels=64, out_channels=64, kernel_size=5, stride=1, padding=2),
            nn.ReLU()
        )

        # Branch 4: 3x3 max pooling followed by 1x1 convolution
        self.branch4 = nn.Sequential(
            nn.MaxPool2D(kernel_size=3, stride=1, padding=1),
            nn.Conv2D(in_channels=3, out_channels=64, kernel_size=1, stride=1, padding=0),
            nn.ReLU()
        )

        # Fully connected layers
        self.fc1 = nn.Linear(in_features=64 * 4 * 32 * 32, out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=10)

    def forward(self, x):
        # Process each branch
        branch1_out = self.branch1(x)
        branch2_out = self.branch2(x)
        branch3_out = self.branch3(x)
        branch4_out = self.branch4(x)

        # Concatenate the outputs along the channel dimension
        concatenated = paddle.concat([branch1_out, branch2_out, branch3_out, branch4_out], axis=1)

        # Flatten the features
        flattened = paddle.flatten(concatenated, start_axis=1)

        # Fully connected layers
        fc1_out = self.fc1(flattened)
        fc1_out = F.relu(fc1_out)
        fc2_out = self.fc2(fc1_out)

        return fc2_out

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn((1, 3, 32, 32))  # Batch size 1, 3 channels, 32x32 image
    output_tensor = model(input_tensor)
    print("Output shape:", output_tensor.shape)