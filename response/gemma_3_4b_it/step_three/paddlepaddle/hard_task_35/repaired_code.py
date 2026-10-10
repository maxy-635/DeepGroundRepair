import paddle
import paddle.nn as nn

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()

        self.conv_block = nn.Layer(
            nn.Conv2D(
                in_channels=3,
                out_channels=16,
                kernel_size=3,
                stride=1,
                padding=1
            )
        )

        self.avg_pool = nn.Layer(
            nn.AdaptiveAvgPool2D(output_size=(1, 1))
        )

        self.fc1 = nn.Layer(
            nn.Linear(16, 128)
        )
        self.fc2 = nn.Layer(
            nn.Linear(128, 10)
        )

        self.dropout = nn.Layer(
            nn.Dropout(p=0.5)
        )

    def forward(self, x):
        x = self.conv_block(x)
        x = self.avg_pool(x)
        x = x.reshape([x.shape[0], 16])

        branch1 = self.fc1(x)
        branch1 = self.dropout(branch1)
        branch1 = self.fc2(branch1)

        branch2 = self.fc1(x)
        branch2 = self.dropout(branch2)
        branch2 = self.fc2(branch2)

        x = paddle.concat([branch1, branch2], axis=1)
        x = nn.Layer(nn.Flatten())(x)
        x = self.fc2(x)

        return x

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn((1, 3, 32, 32))
    output_tensor = model(input_tensor)
    print(output_tensor.shape)