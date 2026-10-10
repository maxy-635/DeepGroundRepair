import paddle
import paddle.nn as nn

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Sequential model definition
        self.feature_extraction = nn.Sequential(
            nn.Conv2D(3, 32, kernel_size=3, stride=1, padding=1),
            nn.AvgPool2D(kernel_size=2, stride=2),
            nn.Conv2D(32, 64, kernel_size=3, stride=1, padding=1),
            nn.AvgPool2D(kernel_size=2, stride=2),
            nn.Conv2D(64, 128, kernel_size=3, stride=1, padding=1)
        )
        
        # Additional convolutional layers and pooling
        self.conv_layers = nn.Sequential(
            nn.Conv2D(128, 256, kernel_size=3, stride=1, padding=1),
            nn.AvgPool2D(kernel_size=2, stride=2)
        )

        self.flatten = nn.Flatten()
        self.fc_layers = nn.Sequential(
            nn.Linear(in_features=200704, out_features=512),
            nn.Dropout(p=0.5),
            nn.Linear(512, 1000)
        )
        self.softmax = nn.Softmax(axis=-1)

    def forward(self, x):
        # the initial shape of x is: (1, 3, 224, 224)
        x = self.feature_extraction(x)  # output tensor shape: (1, 128, 56, 56)
        x = self.conv_layers(x)  # output tensor shape: (1, 256, 28, 28)
        x = self.flatten(x)  # output tensor shape: (1, 200704)
        x = self.fc_layers(x)
        x = self.softmax(x)
        return x

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn(shape=(1, 3, 224, 224))
    output_tensor = model(input_tensor)
    print(output_tensor.shape)