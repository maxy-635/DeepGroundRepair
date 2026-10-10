import paddle
import paddle.nn as nn

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()
        
        # Layers
        self.conv1x1_1 = nn.Conv2D(in_channels=1, out_channels=16, kernel_size=(1, 1), stride=1, padding=0)
        self.dropout1 = nn.Dropout(p=0.1)
        self.conv1x1_2 = nn.Conv2D(in_channels=16, out_channels=32, kernel_size=(1, 1), stride=1, padding=0)
        self.dropout2 = nn.Dropout(p=0.1)
        self.conv3x1 = nn.Conv2D(in_channels=32, out_channels=32, kernel_size=(3, 1), stride=1, padding=0)
        self.dropout3 = nn.Dropout(p=0.1)
        self.conv1x3 = nn.Conv2D(in_channels=32, out_channels=32, kernel_size=(1, 3), stride=1, padding=0)
        self.dropout4 = nn.Dropout(p=0.1)
        self.conv1x1_restore = nn.Conv2D(in_channels=32, out_channels=1, kernel_size=(1, 1), stride=1, padding=0)
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(in_features=784, out_features=10)
        self.relu = nn.ReLU()

    def forward(self, x):
        # Convolutional layers and dropout
        x1 = self.conv1x1_1(x)
        x1 = self.dropout1(x1)
        x2 = self.conv1x1_2(x1)
        x2 = self.dropout2(x2)
        x3 = self.conv3x1(x2)
        x3 = self.dropout3(x3)
        x4 = self.conv1x3(x3)
        x4 = self.dropout4(x4)
        x5 = self.conv1x1_restore(x4)

        # Add processed features to original input
        x = x + x5

        # Flatten and linear layer
        x = self.flatten(x)
        x = self.linear(x)
        
        return x


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn([1, 1, 28, 28])  # Batch size of 1, 1 channel, 28x28 image
    output_tensor = model(input_tensor)
    print(output_tensor.shape)