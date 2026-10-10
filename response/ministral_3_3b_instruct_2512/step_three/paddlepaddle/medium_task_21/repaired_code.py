import paddle
import paddle.nn as nn

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()
        # Branch 1: 1x1 convolution
        self.branch1 = nn.Sequential(
            nn.Conv2D(3, 64, kernel_size=1, stride=1, padding=0),
            nn.Dropout(0.2)
        )

        # Branch 2: 1x1 convolution followed by 3x3 convolution
        self.branch2 = nn.Sequential(
            nn.Conv2D(3, 64, kernel_size=1, stride=1, padding=0),
            nn.Conv2D(64, 64, kernel_size=3, stride=1, padding=1),
            nn.Dropout(0.2)
        )

        # Branch 3: 1x1 convolution followed by two consecutive 3x3 convolutions
        self.branch3 = nn.Sequential(
            nn.Conv2D(3, 64, kernel_size=1, stride=1, padding=0),
            nn.Conv2D(64, 64, kernel_size=3, stride=1, padding=1),
            nn.Conv2D(64, 64, kernel_size=3, stride=1, padding=1),
            nn.Dropout(0.2)
        )

        # Branch 4: Average pooling followed by 1x1 convolution
        self.branch4 = nn.Sequential(
            nn.AvgPool2D(kernel_size=3, stride=1, padding=1),
            nn.Conv2D(3, 64, kernel_size=1, stride=1, padding=0),
            nn.Dropout(0.2)
        )

        # Concatenate all branches
        self.concat = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=256, out_features=512),
            nn.Linear(512, 256),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        # the initial shape of x is: (1, 3, 32, 32)
        branch1_out = self.branch1(x)  # output tensor shape: (1, 64, 32, 32)
        branch2_out = self.branch2(x)  # output tensor shape: (1, 64, 32, 32)
        branch3_out = self.branch3(x)  # output tensor shape: (1, 64, 32, 32)
        branch4_out = self.branch4(x)  # output tensor shape: (1, 64, 32, 32)

        # Concatenate outputs from all branches
        concat_out = paddle.concat([branch1_out, branch2_out, branch3_out, branch4_out], axis=1)  # output tensor shape: (1, 256, 32, 32)

        # Pass through fully connected layers
        out = self.concat(concat_out)
        return out

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn((1, 3, 32, 32))  # Example input tensor with shape (batch_size, channels, height, width)
    output_tensor = model(input_tensor)
    print(output_tensor.shape)