import paddle
import paddle.nn as nn
import paddle.nn.functional as F

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Main path
        self.conv1 = nn.Conv2D(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.dropout1 = nn.Dropout(p=0.2)
        self.conv2 = nn.Conv2D(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.dropout2 = nn.Dropout(p=0.2)
        self.conv3 = nn.Conv2D(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1)

        # Branch path (identity connection)
        self.branch_conv = nn.Conv2D(in_channels=1, out_channels=32, kernel_size=1, stride=1, padding=0)

        # Final layers
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(in_features=32*28*28, out_features=10)  # MNIST has 28x28 images

    def forward(self, x):
        # Main path
        main = self.conv1(x)
        main = self.dropout1(main)
        main = F.relu(main)

        main = self.conv2(main)
        main = self.dropout2(main)
        main = F.relu(main)

        main = self.conv3(main)

        # Branch path
        branch = self.branch_conv(x)

        # Combine paths
        combined = main + branch
        combined = F.relu(combined)

        # Final layers
        out = self.flatten(combined)
        out = self.fc(out)

        return out

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn((1, 1, 28, 28))  # batch_size=1, channels=1, height=28, width=28
    output_tensor = model(input_tensor)
    print("Output shape:", output_tensor.shape)