import paddle
import paddle.nn as nn
import paddle.vision.datasets as datasets

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Main path
        self.main_path = nn.Sequential(
            nn.Conv2D(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2D((1, 1)),
            nn.Flatten(),
            nn.Linear(32 * 1 * 1, 128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features='mask')
        )

        # Branch path
        self.branch_path = nn.Sequential(
            nn.Conv2D(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2D((1, 1)),
            nn.Flatten(),
            nn.Linear(32 * 1 * 1, 128)
        )

    def forward(self, x):
        # the initial shape of x is: (1, 3, 32, 32)
        main_output = self.main_path(x)  # output tensor shape of this layer should be (1, 128)
        branch_output = self.branch_path(x)  # output tensor shape: (1, 128)

        combined_output = paddle.add(main_output, branch_output)

        combined_output = nn.Linear(128, 10)
        output = paddle.nn.Softmax(axis=-1)(combined_output)

        return output

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn((1, 3, 32, 32))
    output_tensor = model(input_tensor)
    print(output_tensor.shape)