import paddle
import paddle.nn as nn

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()
        # Implement the four branches as described in the task
        self.branch1 = nn.Sequential(
            nn.Conv2D(3, 64, kernel_size=1, stride=1, padding=0)
        )
        self.branch2 = nn.Sequential(
            nn.Conv2D(3, 64, kernel_size=1, stride=1, padding=0),
            nn.Conv2D(64, 64, kernel_size=3, stride=1, padding=1)
        )
        self.branch3 = nn.Sequential(
            nn.Conv2D(3, 64, kernel_size=1, stride=1, padding=0),
            nn.Conv2D(64, 64, kernel_size=5, stride=1, padding=2)
        )
        self.branch4 = nn.Sequential(
            nn.MaxPool2D(kernel_size=1, stride=1, padding='same'),
            nn.Conv2D(3, 64, kernel_size=1, stride=1, padding=0)
        )

        # Concatenate the outputs of the four branches
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 4 * 4, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        # the initial shape of x is: (1, 3, 32, 32)
        out1 = self.branch1(x)  # output tensor shape: (1, 64, 32, 32)
        out2 = self.branch2(x)  # output tensor shape: (1, 64, 32, 32)
        out3 = self.branch3(x)  # output tensor shape: (1, 64, 32, 32)
        out4 = self.branch4(x)  # output tensor shape of this layer should be (1, 64, 32, 32)

        # Concatenate the outputs
        out = paddle.concat([out1, out2, out3, out4], axis=1)
        out = self.flatten(out)
        out = paddle.relu(self.fc1(out))
        out = self.fc2(out)
        return out

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn((1, 3, 32, 32))
    output_tensor = model(input_tensor)
    print(output_tensor.shape)