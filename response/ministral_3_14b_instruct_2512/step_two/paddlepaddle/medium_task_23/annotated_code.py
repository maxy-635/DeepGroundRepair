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

        # Path 2: 1x1 followed by 1x7 and 7x1 convolutions
        self.path2 = nn.Sequential(
            nn.Conv2D(in_channels=3, out_channels=64, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2D(in_channels=64, out_channels=64, kernel_size=(1, 7), stride=1, padding=(0, 3)),
            nn.ReLU(),
            nn.Conv2D(in_channels=64, out_channels=64, kernel_size=(7, 1), stride=1, padding=(3, 0)),
            nn.ReLU()
        )

        # Path 3: 1x1 followed by two sets of 1x7 and 7x1 convolutions
        self.path3 = nn.Sequential(
            nn.Conv2D(in_channels=3, out_channels=64, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2D(in_channels=64, out_channels=64, kernel_size=(1, 7), stride=1, padding=(0, 3)),
            nn.ReLU(),
            nn.Conv2D(in_channels=64, out_channels=64, kernel_size=(7, 1), stride=1, padding=(3, 0)),
            nn.ReLU(),
            nn.Conv2D(in_channels=64, out_channels=64, kernel_size=(1, 7), stride=1, padding=(0, 3)),
            nn.ReLU(),
            nn.Conv2D(in_channels=64, out_channels=64, kernel_size=(7, 1), stride=1, padding=(3, 0)),
            nn.ReLU()
        )

        # Path 4: Average pooling followed by 1x1 convolution
        self.path4 = nn.Sequential(
            nn.AdaptiveAvgPool2D(output_size='mask'),
            nn.Conv2D(in_channels=3, out_channels=64, kernel_size=1, stride=1, padding=0),
            nn.ReLU()
        )

        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=64*4*32*32, out_features=10)  # CIFAR-10 has 10 classes
        )

    def forward(self, x):
        # the initial shape of x is: (1, 3, 32, 32)
        # Process each path
        path1_out = self.path1(x)  # output tensor shape: (1, 64, 32, 32)
        path2_out = self.path2(x)  # output tensor shape: (1, 64, 32, 32)
        path3_out = self.path3(x)  # output tensor shape: (1, 64, 32, 32)
        path4_out = self.path4(x)  # output tensor shape of this layer should be (1, 64, 32, 32)

        # Concatenate all paths
        concat_out = paddle.concat([path1_out, path2_out, path3_out, path4_out], axis=1)

        # Final classification
        output = self.fusion(concat_out)
        return output

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn((1, 3, 32, 32))  # Batch size 1, 3 channels, 32x32 image
    output_tensor = model(input_tensor)
    print("Input shape:", input_tensor.shape)
    print("Output shape:", output_tensor.shape)