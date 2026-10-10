import paddle
import paddle.nn as nn

class DL_Model(nn.Sequential):
    def __init__(self):
        super(DL_Model, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2D(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2D(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2D(kernel_size=2, stride=2, padding=0),
            nn.Flatten(),
            nn.Linear(in_features=128, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=10),
            nn.Softmax(axis=-1)
        )
    
    def forward(self, x):
        # the initial shape of x is: (1, 3, 32, 32)
        x = self.layers(x)
        return x


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.rand(shape=[1, 3, 32, 32])
    output_tensor = model(input_tensor)
    print(output_tensor.shape)