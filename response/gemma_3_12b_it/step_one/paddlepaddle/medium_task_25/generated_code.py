import paddle
import paddle.nn as nn
import paddle.nn.functional as F

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Path 1: Single 1x1 convolution
        self.path1 = nn.Sequential(
            nn.Conv2D(in_channels=3, out_channels=64, kernel_size=1, stride=1, padding=0),
            nn.ReLU()
        )

        # Path 2: AvgPool followed by 1x1 convolution
        self.path2 = nn.Sequential(
            nn.AvgPool2D(kernel_size=2, stride=2, padding=0),
            nn.Conv2D(in_channels=3, out_channels=64, kernel_size=1, stride=1, padding=0),
            nn.ReLU()
        )

        # Path 3: 1x1 conv + parallel 1x3 and 3x1 convs
        self.path3 = nn.Sequential(
            nn.Conv2D(in_channels=3, out_channels=64, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2D(in_channels=64, out_channels=64, kernel_size=1, stride=1, padding=0),
            nn.Conv2D(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.Conv2D(in_channels=64, out_channels=64, kernel_size=1, stride=1, padding=0)
        )
        self.path3_concat = nn.Sequential(
            nn.Conv2D(in_channels=64 * 2, out_channels=64, kernel_size=1, stride=1, padding=0)
        )


        # Path 4: 1x1 conv + 3x3 conv + parallel 1x3 and 3x1 convs
        self.path4 = nn.Sequential(
            nn.Conv2D(in_channels=3, out_channels=64, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2D(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2D(in_channels=64, out_channels=64, kernel_size=1, stride=1, padding=0),
            nn.Conv2D(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.Conv2D(in_channels=64, out_channels=64, kernel_size=1, stride=1, padding=0)
        )

        self.path4_concat = nn.Sequential(
            nn.Conv2D(in_channels=64 * 2, out_channels=64, kernel_size=1, stride=1, padding=0)
        )


        # Feature fusion
        self.fusion = nn.Sequential(
            nn.Conv2D(in_channels=64 * 4, out_channels=64, kernel_size=1, stride=1, padding=0)
        )

        # Classification
        self.fc = nn.Linear(in_features=64 * 8 * 8, out_features=10)  # Assuming input size is 32x32, after several conv layers.

    def forward(self, x):
        # Extract features from parallel paths
        path1_out = self.path1(x)
        path2_out = self.path2(x)
        path3_out = self.path3(x)
        path4_out = self.path4(x)

        # Concatenate the outputs of the paths
        x = paddle.concat([path1_out, path2_out, path3_out, path4_out], axis=1)
        x = self.fusion(x)

        # Flatten the feature map
        x = paddle.flatten(x, [1, 2])

        # Classification
        output = self.fc(x)

        return output


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.rand(shape=[1, 3, 32, 32])  # Batch size 1, 3 channels, 32x32 image
    output_tensor = model(input_tensor)
    print(output_tensor.shape)