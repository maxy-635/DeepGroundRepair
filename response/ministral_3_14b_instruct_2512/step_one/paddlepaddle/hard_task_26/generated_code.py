import paddle
import paddle.nn as nn
import paddle.nn.functional as F

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Main path
        self.main_path = nn.Sequential(
            nn.Conv2D(in_channels=3, out_channels=64, kernel_size=1, stride=1, padding=0),
            nn.ReLU()
        )

        # Branch 1
        self.branch1 = nn.Sequential(
            nn.Conv2D(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )

        # Branch 2
        self.branch2 = nn.Sequential(
            nn.MaxPool2D(kernel_size=2, stride=2, padding=0),
            nn.Conv2D(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest')
        )

        # Branch 3
        self.branch3 = nn.Sequential(
            nn.MaxPool2D(kernel_size=2, stride=2, padding=0),
            nn.Conv2D(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest')
        )

        # Combine branches
        self.combine = nn.Sequential(
            nn.Conv2D(in_channels=192, out_channels=64, kernel_size=1, stride=1, padding=0),
            nn.ReLU()
        )

        # Branch path
        self.branch_path = nn.Sequential(
            nn.Conv2D(in_channels=3, out_channels=64, kernel_size=1, stride=1, padding=0),
            nn.ReLU()
        )

        # Classification layers
        self.classifier = nn.Sequential(
            nn.Linear(in_features=64 * 32 * 32, out_features=512),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=512, out_features=10)
        )

    def forward(self, x):
        # Main path
        main_out = self.main_path(x)

        # Branch 1
        branch1_out = self.branch1(main_out)

        # Branch 2
        branch2_out = self.branch2(main_out)

        # Branch 3
        branch3_out = self.branch3(main_out)

        # Combine branches
        combined = paddle.concat([branch1_out, branch2_out, branch3_out], axis=1)
        combined = self.combine(combined)

        # Add branch path
        branch_out = self.branch_path(x)
        combined = combined + branch_out

        # Flatten and classify
        combined = paddle.flatten(combined, start_axis=1)
        output = self.classifier(combined)

        return output

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn((1, 3, 32, 32))  # Batch size 1, 3 channels, 32x32 image
    output_tensor = model(input_tensor)
    print("Input shape:", input_tensor.shape)
    print("Output shape:", output_tensor.shape)