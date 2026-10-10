import paddle
import paddle.nn as nn
import paddle.nn.functional as F

class DL_Model(nn.Layer):
    def __init__(self, num_classes=10):
        super(DL_Model, self).__init__()

        # Branch 1: 3x3 convolutions
        self.branch1 = nn.Sequential(
            nn.Conv2D(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )

        # Branch 2: 1x1 convolutions followed by two 3x3 convolutions
        self.branch2 = nn.Sequential(
            nn.Conv2D(in_channels=3, out_channels=64, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2D(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2D(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )

        # Branch 3: Max pooling
        self.branch3 = nn.Sequential(
            nn.MaxPool2D(kernel_size=3, stride=1, padding=1),
            nn.Conv2D(in_channels=3, out_channels=64, kernel_size=1, stride=1, padding=0),
            nn.ReLU()
        )

        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=64 * 32 * 32 * 3, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=num_classes)
        )

    def forward(self, x):
        # Process each branch
        out1 = self.branch1(x)
        out2 = self.branch2(x)
        out3 = self.branch3(x)

        # Concatenate the outputs
        out = paddle.concat([out1, out2, out3], axis=1)

        # Pass through fully connected layers
        out = self.fc(out)

        return out

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn([1, 3, 32, 32])  # Batch size of 1, 3 channels, 32x32 image
    output_tensor = model(input_tensor)
    print("Output shape:", output_tensor.shape)