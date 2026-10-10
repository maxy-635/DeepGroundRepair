import paddle
import paddle.nn as nn
import paddle.nn.functional as F

class DL_Model(nn.Sequential):
    def __init__(self):
        super().__init__(
            nn.Sequential(
                nn.Conv2D(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1),
                nn.Conv2D(in_channels=16, out_channels=16, kernel_size=1, stride=1, padding=0),
                nn.Conv2D(in_channels=16, out_channels=16, kernel_size=1, stride=1, padding=0),
                nn.AvgPool2D(kernel_size=2, stride=2, padding=0),
                nn.Dropout(p=0.25),
            ),
            nn.Sequential(
                nn.Conv2D(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
                nn.Conv2D(in_channels=32, out_channels=32, kernel_size=1, stride=1, padding=0),
                nn.Conv2D(in_channels=32, out_channels=32, kernel_size=1, stride=1, padding=0),
                nn.AvgPool2D(kernel_size=2, stride=2, padding=0),
                nn.Dropout(p=0.25),
            ),
            nn.AdaptiveAvgPool2D(output_size=1),
            nn.Flatten(),
            nn.Linear(in_features=32, out_features=10)
        )

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn(shape=[1, 1, 28, 28])
    output_tensor = model(input_tensor)
    print(output_tensor.shape)