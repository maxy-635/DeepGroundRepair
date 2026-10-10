import paddle
import paddle.nn as nn

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()
        # Path 1: Single 1x1 convolution
        self.path1 = nn.Sequential(
            nn.Conv2D(in_channels=3, out_channels=64, kernel_size=1, stride=1, padding=0),
            nn.PReLU(num_parameters=64)
        )

        # Path 2: Average pooling followed by 1x1 convolution
        self.path2 = nn.Sequential(
            nn.AvgPool2D(kernel_size=3, stride=1, padding=1),
            nn.Conv2D(in_channels=3, out_channels=64, kernel_size=1, stride=1, padding=0),
            nn.PReLU(num_parameters=64)
        )

        # Path 3: 1x1 convolution followed by two parallel 1x3 and 3x1 convolutions
        self.path3 = nn.Sequential(
            nn.Conv2D(in_channels=3, out_channels=64, kernel_size=1, stride=1, padding=0),
            nn.Sequential(
                nn.Conv2D(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
                nn.Conv2D(in_channels=64, out_channels=64, kernel_size=1, stride=1, padding=0),
                nn.PReLU(num_parameters=64)
            ),
            nn.Conv2D(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.Conv2D(in_channels=64, out_channels=64, kernel_size=1, stride=1, padding=0),
            nn.PReLU(num_parameters=64)
        )

        # Path 4: 1x1 convolution followed by 3x3 convolution, then two parallel 1x3 and 3x1 convolutions
        self.path4 = nn.Sequential(
            nn.Conv2D(in_channels=3, out_channels=64, kernel_size=1, stride=1, padding=0),
            nn.Conv2D(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.Sequential(
                nn.Conv2D(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
                nn.Conv2D(in_channels=64, out_channels=64, kernel_size=1, stride=1, padding=0),
                nn.PReLU(num_parameters=64)
            ),
            nn.Conv2D(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.Conv2D(in_channels=64, out_channels=64, kernel_size=1, stride=1, padding=0),
            nn.PReLU(num_parameters=64)
        )

        # Concatenate the outputs of the four paths
        self.concat = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=64 * 32 * 32, out_features=1024),
            nn.PReLU(num_parameters=1024),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=1024, out_features=10)
        )

    def forward(self, x):
        path1_out = self.path1(x)
        path2_out = self.path2(x)
        path3_out = self.path3(x)
        path4_out = self.path4(x)

        # Concatenate the outputs of the four paths
        concat_out = paddle.concat([path1_out, path2_out, path3_out, path4_out], axis=1)
        output = self.concat(concat_out)
        return output

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn((1, 3, 32, 32))  # Example input tensor with shape (batch_size, channels, height, width)
    output_tensor = model(input_tensor)
    print(output_tensor.shape)