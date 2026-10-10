import paddle
import paddle.nn as nn
import paddle.nn.functional as F

class DL_Model(nn.Layer):
    def __init__(self):
        super().__init__()
        # Main Path
        self.main_path = nn.Sequential(
            nn.Conv2D(in_channels=3, out_channels=32, kernel_size=1, stride=1, padding=0),
            nn.Sequential(
                nn.Conv2D(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            ),
            nn.Sequential(
                nn.MaxPool2D(kernel_size=2, stride=2, padding=0),
                nn.Conv2D(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
                nn.Upsample(size=(32, 32), scale_factor=2, mode='nearest', align_corners=False)
            ),
            nn.Sequential(
                nn.MaxPool2D(kernel_size=2, stride=2, padding=0),
                nn.Conv2D(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
                nn.Upsample(size=(32, 32), scale_factor=2, mode='nearest', align_corners=False)
            ),
            nn.Conv2D(in_channels=192, out_channels=64, kernel_size=1, stride=1, padding=0)
        )

        # Branch Path
        self.branch_path = nn.Sequential(
            nn.Conv2D(in_channels=3, out_channels=32, kernel_size=1, stride=1, padding=0),
            nn.Conv2D(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        )

        # Classification Layers
        self.classifier = nn.Sequential(
            nn.Linear(in_features=64, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=10)
        )

    def forward(self, x):
        # Main Path
        main_path_output = self.main_path(x)

        # Branch Path
        branch_path_output = self.branch_path(x)

        # Add the outputs of both paths
        combined_output = main_path_output + branch_path_output

        # Classification
        classification_output = self.classifier(combined_output.flatten())

        return classification_output


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn(shape=(1, 3, 32, 32))  # Batch size of 1, 3 channels, 32x32 image
    output_tensor = model(input_tensor)
    print(output_tensor.shape)