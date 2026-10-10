import paddle
import paddle.nn as nn
import paddle.nn.functional as F

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()
        # Block 1: Three parallel paths with average pooling of different scales
        self.pool1x1 = nn.AvgPool2D(kernel_size='mask', stride='mask', padding='mask')
        self.pool2x2 = nn.AvgPool2D(kernel_size=2, stride=2, padding=0)
        self.pool4x4 = nn.AvgPool2D(kernel_size=4, stride=4, padding=0)

        # Dropout layers for regularization
        self.dropout1 = nn.Dropout(p=0.5)
        self.dropout2 = nn.Dropout(p=0.5)
        self.dropout3 = nn.Dropout(p=0.5)

        # Concatenation layer for block 1 output
        self.concat_block1 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28, 1024),  # Assuming input is 28x28
            nn.ReLU()
        )

        # Block 2: Multiple branch connections
        self.conv1x1_1 = nn.Conv2D(in_channels=1024, out_channels=256, kernel_size=1, stride=1, padding=0)
        self.conv1x1_2 = nn.Conv2D(in_channels=1024, out_channels=256, kernel_size=1, stride=1, padding=0)
        self.conv3x3_1 = nn.Conv2D(in_channels=1024, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.conv3x3_2 = nn.Conv2D(in_channels=1024, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.conv3x3_3 = nn.Conv2D(in_channels=1024, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.pool1x1_2 = nn.AvgPool2D(kernel_size=1, stride=1, padding=0)
        self.conv1x1_3 = nn.Conv2D(in_channels=1024, out_channels=256, kernel_size=1, stride=1, padding=0)

        # Concatenation layer for block 2 output
        self.concat_block2 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256*4*4, 1024),  # Assuming output is 4x4
            nn.ReLU()
        )

        # Final classification layers
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        # the initial shape of x is: (1, 1, 28, 28)
        # Block 1: Three parallel paths
        pool1x1_out = self.pool1x1(x)  # output tensor shape of this layer should be (1, 1, 14, 14)
        pool2x2_out = self.pool2x2(x)  # output tensor shape: (1, 1, 14, 14)
        pool4x4_out = self.pool4x4(x)  # output tensor shape: (1, 1, 7, 7)

        # Apply dropout and flatten
        dropout1_out = self.dropout1(pool1x1_out)
        dropout2_out = self.dropout2(pool2x2_out)  # output tensor shape: (1, 1, 14, 14)
        dropout3_out = self.dropout3(pool4x4_out)  # output tensor shape: (1, 1, 7, 7)

        # Concatenate outputs
        block1_out = paddle.concat([dropout1_out, dropout2_out, dropout3_out], axis=1)
        block1_out = self.concat_block1(block1_out)

        # Reshape for block 2
        reshaped_block1 = paddle.reshape(block1_out, [block1_out.shape[0], 1024, 4, 4])

        # Block 2: Multiple branch connections
        conv1x1_1_out = self.conv1x1_1(reshaped_block1)
        conv1x1_2_out = self.conv1x1_2(reshaped_block1)
        conv3x3_1_out = self.conv3x3_1(reshaped_block1)
        conv3x3_2_out = self.conv3x3_2(conv3x3_1_out)
        conv3x3_3_out = self.conv3x3_3(conv3x3_2_out)
        pool1x1_2_out = self.pool1x1_2(reshaped_block1)
        conv1x1_3_out = self.conv1x1_3(pool1x1_2_out)

        # Concatenate outputs
        block2_out = paddle.concat([conv1x1_1_out, conv1x1_2_out, conv3x3_3_out, conv1x1_3_out], axis=1)
        block2_out = self.concat_block2(block2_out)

        # Final classification
        fc1_out = F.relu(self.fc1(block2_out))
        fc2_out = self.fc2(fc1_out)

        return fc2_out

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn((1, 1, 28, 28))  # Dummy input tensor
    output_tensor = model(input_tensor)
    print(output_tensor.shape)