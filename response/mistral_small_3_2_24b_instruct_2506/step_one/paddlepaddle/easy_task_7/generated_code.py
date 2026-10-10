import paddle
import paddle.nn as nn

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Main path
        self.conv1 = nn.Conv2D(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.dropout1 = nn.Dropout(p=0.25)
        self.conv2 = nn.Conv2D(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.dropout2 = nn.Dropout(p=0.25)
        self.conv3 = nn.Conv2D(in_channels=64, out_channels=1, kernel_size=3, stride=1, padding=1)

        # Branch path (identity connection)
        self.branch = nn.Sequential()

        # Final layers
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(in_features=28*28, out_features=10)

    def forward(self, x):
        # Main path
        out = self.conv1(x)
        out = self.dropout1(out)
        out = self.conv2(out)
        out = self.dropout2(out)
        out = self.conv3(out)

        # Branch path
        branch_out = self.branch(x)

        # Combine paths
        combined = paddle.add(out, branch_out)

        # Final layers
        out = self.flatten(combined)
        out = self.fc(out)

        return out

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn(shape=[1, 1, 28, 28])
    output_tensor = model(input_tensor)
    print(output_tensor.shape)