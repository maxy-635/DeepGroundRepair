import paddle
import paddle.nn as nn
import numpy as np

class DL_Model(nn.Sequential):
    def __init__(self):
        super().__init__(
            nn.Sequential(
                nn.Sequential(
                    nn.MaxPool2D(kernel_size=1, stride=1, padding=0),
                    nn.Flatten(),
                    nn.Dropout(p=0.2)
                ),
                nn.Sequential(
                    nn.MaxPool2D(kernel_size=2, stride=2, padding=0),
                    nn.Flatten(),
                    nn.Dropout(p=0.2)
                ),
                nn.Sequential(
                    nn.MaxPool2D(kernel_size=4, stride=4, padding=0),
                    nn.Flatten(),
                    nn.Dropout(p=0.2)
                )
            ),
            nn.Linear(in_features=3 * (2**2 + 2**1 + 2**0), out_features=128),
            paddle.reshape(shape=(-1, 4, 128), x=paddle.reshape(shape=(-1, 4, 128))),
            nn.Sequential(
                nn.Sequential(
                    nn.Conv2D(in_channels=4, out_channels=32, kernel_size=1, stride=1, padding=0),
                ),
                nn.Sequential(
                    nn.Conv2D(in_channels=32, out_channels=64, kernel_size=1, stride=1, padding=0),
                    nn.Conv2D(in_channels=64, out_channels=64, kernel_size=7, stride=1, padding=3),
                    nn.Conv2D(in_channels=64, out_channels=64, kernel_size=7, stride=1, padding=3)
                ),
                nn.Sequential(
                    nn.Conv2D(in_channels=64, out_channels=64, kernel_size=1, stride=1, padding=0),
                    nn.Conv2D(in_channels=64, out_channels=64, kernel_size=7, stride=1, padding=3),
                    nn.Conv2D(in_channels=64, out_channels=64, kernel_size=7, stride=1, padding=3)
                ),
                nn.Sequential(
                    nn.AvgPool2D(kernel_size=2, stride=2, padding=0),
                    nn.Conv2D(in_channels=64, out_channels=64, kernel_size=1, stride=1, padding=0)
                )
            ),
            nn.Flatten(),
            nn.Linear(in_features=64 * 4, out_features=128),
            nn.Linear(in_features=128, out_features=10)
        )

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.to_tensor(data=np.random.rand(1, 1, 28, 28), dtype='float32')
    output_tensor = model(input_tensor)
    print(output_tensor.shape)