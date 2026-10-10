import paddle
import paddle.nn as nn

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()
        
        # First block
        self.pool1 = nn.MaxPool2D(kernel_size=1, stride=1, padding=0)
        self.pool2 = nn.MaxPool2D(kernel_size=2, stride=2, padding=0)
        self.pool3 = nn.MaxPool2D(kernel_size=4, stride=4, padding=0)
        self.flatten1 = nn.Flatten()
        self.flatten2 = nn.Flatten()
        self.flatten3 = nn.Flatten()
        self.dropout = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(in_features=16*16*3 + 4*4*3 + 1*1*3, out_features=64)
        self.reshape = nn.Reshape(shape=[64, 1, 1, 1])

        # Second block
        self.conv1 = nn.Conv2D(in_channels=64, out_channels=32, kernel_size=1, stride=1, padding=0, groups=4)
        self.conv2 = nn.Conv2D(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1, groups=4)
        self.conv3 = nn.Conv2D(in_channels=64, out_channels=32, kernel_size=5, stride=1, padding=2, groups=4)
        self.conv4 = nn.Conv2D(in_channels=64, out_channels=32, kernel_size=7, stride=1, padding=3, groups=4)
        self.flatten4 = nn.Flatten()
        self.fc2 = nn.Linear(in_features=32 * 4 * 4, out_features=10)

    def forward(self, x):
        # First block
        x1 = self.pool1(x)
        x2 = self.pool2(x)
        x3 = self.pool3(x)
        
        x1_flat = self.flatten1(x1)
        x2_flat = self.flatten2(x2)
        x3_flat = self.flatten3(x3)

        x_concat = self.dropout(paddle.concat([x1_flat, x2_flat, x3_flat], axis=1))
        x_fc = self.fc1(x_concat)
        x_reshape = self.reshape(x_fc)
        
        # Second block
        x_group1 = self.conv1(x_reshape)
        x_group2 = self.conv2(x_reshape)
        x_group3 = self.conv3(x_reshape)
        x_group4 = self.conv4(x_reshape)
        
        x_concat2 = paddle.concat([x_group1, x_group2, x_group3, x_group4], axis=1)
        x_flat = self.flatten4(x_concat2)
        output = self.fc2(x_flat)
        
        return output

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn(shape=[1, 3, 32, 32])
    output_tensor = model(input_tensor)
    print(output_tensor.shape)