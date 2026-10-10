import paddle
import paddle.nn as nn

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Block 1: Three parallel paths with different pooling scales
        self.pool1 = nn.AvgPool2D(kernel_size=1, stride=1, padding=0)
        self.pool2 = nn.AvgPool2D(kernel_size=2, stride=2, padding=0)
        self.pool3 = nn.AvgPool2D(kernel_size=4, stride=4, padding=0)

        self.dropout1 = nn.Dropout(p=0.5)
        self.dropout2 = nn.Dropout(p=0.5)
        self.dropout3 = nn.Dropout(p=0.5)

        self.flatten = nn.Flatten()

        # Fully connected layer and reshaping between Block 1 and Block 2
        self.fc1 = nn.Linear(in_features=784, out_features=128)
        self.reshape = paddle.reshape

        # Block 2: Multiple branch connections
        self.branch1_conv1 = nn.Conv2D(in_channels=1, out_channels=32, kernel_size=1, stride=1, padding=0)
        self.branch2_conv1 = nn.Conv2D(in_channels=1, out_channels=32, kernel_size=1, stride=1, padding=0)
        self.branch2_conv2 = nn.Conv2D(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.branch3_conv1 = nn.Conv2D(in_channels=1, out_channels=32, kernel_size=1, stride=1, padding=0)
        self.branch3_conv2 = nn.Conv2D(in_channels=32, out_features=32, kernel_size=3, stride=1, padding=1)
        self.branch3_conv3 = nn.Conv2D(in_channels=32, out_features=32, kernel_size=3, stride=1, padding=1)
        self.branch4_pool = nn.AvgPool2D(kernel_size=3, stride=1, padding=1)
        self.branch4_conv = nn.Conv2D(in_channels=1, out_channels=32, kernel_size=1, stride=1, padding=0)

        # Final fully connected layers
        self.fc2 = nn.Linear(in_features=128, out_features=10)

    def forward(self, x):
        # Block 1 processing
        pool1_out = self.pool1(x)
        pool2_out = self.pool2(x)
        pool3_out = self.pool3(x)

        pool1_flat = self.flatten(pool1_out)
        pool2_flat = self.flatten(pool2_out)
        pool3_flat = self.flatten(pool3_out)

        pool1_drop = self.dropout1(pool1_flat)
        pool2_drop = self.dropout2(pool2_flat)
        pool3_drop = self.dropout3(pool3_flat)

        block1_out = paddle.concat([pool1_drop, pool2_drop, pool3_drop], axis=1)

        # Between Block 1 and Block 2
        fc1_out = self.fc1(block1_out)
        reshaped_out = self.reshape(fc1_out, shape=[-1, 1, 8, 2])  # Example reshaping

        # Block 2 processing
        branch1_out = self.branch1_conv1(reshaped_out)
        branch2_out = self.branch2_conv2(self.branch2_conv1(reshaped_out))
        branch3_out = self.branch3_conv3(self.branch3_conv2(self.branch3_conv1(reshaped_out)))
        branch4_out = self.branch4_conv(self.branch4_pool(reshaped_out))

        block2_out = paddle.concat([branch1_out, branch2_out, branch3_out, branch4_out], axis=1)

        # Final classification
        flat_out = self.flatten(block2_out)
        fc2_out = self.fc2(flat_out)

        return fc2_out

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn(shape=[1, 1, 28, 28])  # MNIST image shape
    output_tensor = model(input_tensor)
    print(output_tensor.shape)