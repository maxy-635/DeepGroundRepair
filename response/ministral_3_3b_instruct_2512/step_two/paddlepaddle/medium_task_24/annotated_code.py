import paddle
import paddle.nn as nn

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()
        # Branch 1: <1x1 convolution, 3x3 convolution>
        self.branch1 = nn.Sequential(
            nn.Conv2D(3, 64, kernel_size=1, stride=1, padding=0),
            nn.Conv2D(64, 64, kernel_size=3, stride=1, padding=1),
            nn.Dropout(0.5)
        )

        # Branch 2: <1x1 convolution, 1x7 convolution, 7x1 convolution, 3x3 convolution>
        self.branch2 = nn.Sequential(
            nn.Conv2D(3, 64, kernel_size=1, stride=1, padding=0),
            nn.Conv2D(64, 64, kernel_size=7, stride=1, padding=3),
            nn.Conv2D(64, 64, kernel_size=1, stride=1, padding=0),
            nn.Conv2D(64, 64, kernel_size=3, stride=1, padding=1),
            nn.Dropout(0.5)
        )

        # Branch 3: <max pooling>
        self.branch3 = nn.Sequential(
            nn.MaxPool2D(kernel_size='mask', stride='mask', padding='mask'),
            nn.Dropout(0.5)
        )

        # Concatenate branches
        self.concat = nn.Flatten()
        self.fc1 = nn.Linear(64 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 10)

    def forward(self, x):
        # the initial shape of x is: (1, 3, 32, 32)
        branch1_out = self.branch1(x)  # output tensor shape: (1, 64, 32, 32)
        branch2_out = self.branch2(x)  # output tensor shape: (1, 64, 32, 32)
        branch3_out = self.branch3(x)  # output tensor shape of this layer should be (1, 3, 32, 32)

        # Concatenate outputs from all branches
        concat_out = paddle.concat([branch1_out, branch2_out, branch3_out], axis=1)
        concat_out = self.concat(concat_out)

        # Fully connected layers
        x = paddle.nn.functional.relu(self.fc1(concat_out))
        x = paddle.nn.functional.relu(self.fc2(x))
        x = self.fc3(x)
        return x

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn((1, 3, 32, 32))  # Example input tensor with shape (batch_size, channels, height, width)
    output_tensor = model(input_tensor)
    print(output_tensor.shape)