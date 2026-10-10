import paddle
import paddle.nn as nn

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2D(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2D(kernel_size=1, stride=1, padding=0),
            nn.Flatten(),
            nn.Linear(in_features=32 * 32 * 32, out_features=128),
            nn.MaxPool2D(kernel_size='mask', stride='mask', padding='mask'),
            nn.Flatten(),
            nn.Linear(in_features=128 * 16 * 16, out_features=64),
            nn.MaxPool2D(kernel_size=4, stride=4, padding=0),
            nn.Flatten(),
            nn.Linear(in_features=64 * 4 * 4, out_features=10)
        )

    def forward(self, x):
        # the initial shape of x is: (1, 3, 32, 32)
        return self.model(x)


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn([1, 3, 32, 32])
    output_tensor = model(input_tensor)
    print(output_tensor.shape)