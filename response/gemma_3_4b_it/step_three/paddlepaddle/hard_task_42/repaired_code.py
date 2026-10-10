import paddle
import paddle.nn as nn
import paddle.io
import paddle.distributed.elastic as elastic

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Block 1
        self.conv1_1 = nn.Conv2D(in_channels=1, out_channels=32, kernel_size=1, stride=1, padding=0)
        self.pool1_1 = nn.MaxPool2D(kernel_size=1, stride=1, padding=0)
        self.conv1_2 = nn.Conv2D(in_channels=32, out_channels=64, kernel_size=2, stride=2, padding=0)
        self.pool1_2 = nn.MaxPool2D(kernel_size=2, stride=2, padding=0)
        self.conv1_3 = nn.Conv2D(in_channels=64, out_channels=128, kernel_size=4, stride=4, padding=0)
        self.pool1_3 = nn.MaxPool2D(kernel_size=4, stride=4, padding=0)

        self.flatten1 = nn.Flatten()
        self.dropout1 = nn.Dropout(p=0.5)

        # Fully connected layer and reshaping
        self.fc1 = nn.Linear(in_features=128 * 32 * 32, out_features=512)
        self.reshape = paddle.nn.ReshapeTransform(in_event_shape='full')


        # Block 2
        self.conv2_1 = nn.Conv2D(in_channels=128, out_channels=128, kernel_size=1, stride=1, padding=0)
        self.pool2_1 = nn.MaxPool2D(kernel_size=1, stride=1, padding=0)
        self.conv2_2 = nn.Conv2D(in_channels=128, out_channels=256, kernel_size=1, stride=1, padding=0)
        self.pool2_2 = nn.MaxPool2D(kernel_size=1, stride=1, padding=0)
        self.conv2_3 = nn.Conv2D(in_channels=256, out_channels=512, kernel_size=1, stride=1, padding=0)
        self.pool2_3 = nn.MaxPool2D(kernel_size=1, stride=1, padding=0)
        self.avgpool2_1 = nn.AvgPool2D(kernel_size=1, stride=1, padding=0)
        self.conv2_4 = nn.Conv2D(in_channels=512, out_channels=512, kernel_size=1, stride=1, padding=0)

        self.concat2 = nn.Concat(dim=1)

        # Final fully connected layers
        self.fc2 = nn.Linear(in_features=512 * 512, out_features=10)
        self.fc3 = nn.Linear(in_features=10, out_features=10)


    def forward(self, x):
        # Block 1
        x = self.conv1_1(x)
        x = self.pool1_1(x)
        x = self.conv1_2(x)
        x = self.pool1_2(x)
        x = self.conv1_3(x)
        x = self.pool1_3(x)

        x = self.flatten1(x)
        x = self.dropout1(x)

        # Fully connected layer and reshaping
        x = self.fc1(x)
        x = self.reshape(x)

        # Block 2
        x = self.conv2_1(x)
        x = self.pool2_1(x)
        x = self.conv2_2(x)
        x = self.pool2_2(x)
        x = self.conv2_3(x)
        x = self.pool2_3(x)
        x = self.avgpool2_1(x)
        x = self.conv2_4(x)

        x = self.concat2([x, x, x, x])

        # Final classification
        x = self.fc2(x)
        x = self.fc3(x)

        return x

if __name__ == "__main__":
    model = DL_Model()
    # Dummy input tensor
    dummy_input = paddle.randn(batch_size=1, channels=1, height=28, width=28, dtype=paddle.float32)
    output_tensor = model(dummy_input)
    print("Output shape:", output_tensor.shape)