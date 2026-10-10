import paddle
import paddle.nn as nn
import paddle.nn.functional as F

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Block 1
        self.pool1_1 = nn.AvgPool2D(kernel_size=1, stride=1, padding=0)
        self.pool1_2 = nn.AvgPool2D(kernel_size=2, stride=2, padding=0)
        self.pool1_4 = nn.AvgPool2D(kernel_size=4, stride=4, padding=0)
        self.flatten1_1 = nn.Flatten()
        self.flatten1_2 = nn.Flatten()
        self.flatten1_4 = nn.Flatten()
        self.dropout1 = nn.Dropout(p=0.5)

        # FC layer and reshape
        self.fc1 = nn.Linear(in_features=1029, out_features=64)
        
        # Block 2
        self.conv2_1 = nn.Conv2D(in_channels=64, out_channels=32, kernel_size=1, stride=1, padding=0)
        self.conv2_2_1 = nn.Conv2D(in_channels=64, out_channels=32, kernel_size=1, stride=1, padding=0)
        self.conv2_2_2 = nn.Conv2D(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2_3_1 = nn.Conv2D(in_channels=64, out_channels=32, kernel_size=1, stride=1, padding=0)
        self.conv2_3_2 = nn.Conv2D(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2_3_3 = nn.Conv2D(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.pool2_1 = nn.AdaptiveAvgPool2D(output_size=(1, 1))
        self.conv2_4 = nn.Conv2D(in_channels=64, out_channels=32, kernel_size=1, stride=1, padding=0)

        # Final FC layers
        self.fc2 = nn.Linear(in_features=32*4, out_features=128)
        self.relu = nn.ReLU()
        self.fc3 = nn.Linear(in_features=128, out_features=10)

    def forward(self, x):
        # the initial shape of x is: (1, 1, 28, 28)
        # Block 1
        out1_1 = self.pool1_1(x)  # output tensor shape: (1, 1, 28, 28)
        out1_2 = self.pool1_2(x)  # output tensor shape: (1, 1, 14, 14)
        out1_4 = self.pool1_4(x)  # output tensor shape: (1, 1, 7, 7)

        out1_1 = self.flatten1_1(out1_1)  # output tensor shape: (1, 784)
        out1_2 = self.flatten1_2(out1_2)  # output tensor shape: (1, 196)
        out1_4 = self.flatten1_4(out1_4)  # output tensor shape: (1, 49)
        
        out1_1 = self.dropout1(out1_1)  # output tensor shape: (1, 784)
        out1_2 = self.dropout1(out1_2)  # output tensor shape: (1, 196)
        out1_4 = self.dropout1(out1_4)  # output tensor shape: (1, 49)

        out1 = paddle.concat([out1_1, out1_2, out1_4], axis=1)  # output tensor shape: (1, 1029)

        # FC layer and reshape
        out1 = self.fc1(out1)
        out1 = out1.reshape([out1.shape[0], 64, 1, 1])

        # Block 2
        out2_1 = self.conv2_1(out1)
        out2_2_1 = self.conv2_2_1(out1)
        out2_2_2 = self.conv2_2_2(out2_2_1)
        out2_3_1 = self.conv2_3_1(out1)
        out2_3_2 = self.conv2_3_2(out2_3_1)
        out2_3_3 = self.conv2_3_3(out2_3_2)
        out2_4 = self.pool2_1(out1)
        out2_4 = self.conv2_4(out2_4)

        out2 = paddle.concat([out2_1, out2_2_2, out2_3_3, out2_4], axis=1)

        # Final FC layers
        out2 = out2.reshape([out2.shape[0], 32*4])
        out2 = self.fc2(out2)
        out2 = self.relu(out2)
        out2 = self.fc3(out2)

        return out2


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn([1, 1, 28, 28])  # Example input tensor
    output_tensor = model(input_tensor)
    print(output_tensor.shape) # Expected output shape: [1, 10]