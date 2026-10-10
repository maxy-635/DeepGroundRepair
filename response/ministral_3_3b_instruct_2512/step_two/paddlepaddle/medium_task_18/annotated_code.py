import paddle
import paddle.nn as nn

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()
        # Define layers
        self.conv1 = nn.Conv2D(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.layer_norm1 = nn.LayerNorm(normalized_shape='mask')
        self.conv2 = nn.Conv2D(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.layer_norm2 = nn.LayerNorm(normalized_shape=128)
        self.pool1 = nn.MaxPool2D(kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2D(in_channels=128, out_channels=256, kernel_size=5, stride=1, padding=2)
        self.layer_norm3 = nn.LayerNorm(normalized_shape=256)
        self.conv4 = nn.Conv2D(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.layer_norm4 = nn.LayerNorm(normalized_shape=512)
        self.pool2 = nn.MaxPool2D(kernel_size=3, stride=2, padding=1)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_features=512 * 4 * 4, out_features=1024)
        self.fc2 = nn.Linear(in_features=1024, out_features=10)

    def forward(self, x):
        # the initial shape of x is: (1, 3, 32, 32)
        x = self.conv1(x)  # output tensor shape: (1, 64, 32, 32)
        x = self.layer_norm1(x)
        x = nn.ReLU()(x)
        x = self.conv2(x)
        x = self.layer_norm2(x)
        x = nn.ReLU()(x)
        x = self.pool1(x)
        x = self.conv3(x)
        x = self.layer_norm3(x)
        x = nn.ReLU()(x)
        x = self.conv4(x)
        x = self.layer_norm4(x)
        x = nn.ReLU()(self.pool2(x))
        x = self.flatten(x)
        x = self.fc1(x)
        x = nn.ReLU()(x)
        x = self.fc2(x)
        return x

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn((1, 3, 32, 32))
    output_tensor = model(input_tensor)
    print(output_tensor.shape)