import paddle
import paddle.nn as nn
import paddle.nn.functional as F

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()
        # Define the layers
        self.conv1_1 = nn.Conv1D(
            in_channels=1,
            out_channels=32,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1
        )
        self.dropout1 = F.dropout(x='mask', p='mask', training='mask')
        self.conv3_1 = nn.Conv1D(
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=1
        )
        self.dropout2 = F.dropout(x=None, p=0.25, training=True)  # Placeholder
        self.conv1_3 = nn.Conv1D(
            in_channels=64,
            out_channels=32,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1
        )
        self.dropout3 = F.dropout(x=None, p=0.25, training=True)  # Placeholder
        self.conv1_1_restore = nn.Conv1D(
            in_channels=32,
            out_channels=1,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1
        )
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(in_features=32 * 28 * 28, out_features=10)  # Assuming 28x28 input images

    def forward(self, x):
        # Pass through the layers
        x = self.conv1_1(x)
        x = self.dropout1(x)
        x = self.conv3_1(x)
        x = self.dropout2(x)
        x = self.conv1_3(x)
        x = self.dropout3(x)
        x = self.conv1_1_restore(x)
        
        # Add the original input
        x = paddle.add(x, x)
        
        x = self.flatten(x)
        x = self.fc(x)
        return x


if __name__ == "__main__":
    model = DL_Model()
    # Dummy input tensor (batch size 1, 1 channel, 28x28 image)
    input_tensor = paddle.randn(shape=[1, 1, 28, 28])
    output_tensor = model(input_tensor)
    print(output_tensor.shape)