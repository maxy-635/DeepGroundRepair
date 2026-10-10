import paddle
import paddle.nn as nn

class DL_Model(nn.Sequential):
    def __init__(self):
        super(DL_Model, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2D(in_channels=1, out_channels=32, kernel_size=1, stride=2, padding=0),
            nn.Conv2D(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1),
            nn.Conv2D(in_channels=64, out_channels=128, kernel_size=1, stride=2, padding=0),
            nn.Flatten(),
            nn.Linear(in_features='mask', out_features=10)
        )
    
    def forward(self, x):
        # the initial shape of x is: (1, 1, 28, 28)
        x = self.layers(x)
        return x

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.rand(shape=(1, 1, 28, 28))
    output_tensor = model(input_tensor)
    print(output_tensor.shape)