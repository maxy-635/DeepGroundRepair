import paddle
import paddle.nn as nn
import paddle.nn.functional as F

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Block 1: Parallel Average Pooling
        self.pool1x1 = nn.AvgPool2D(kernel_size=1, stride=1, padding=0)
        self.pool2x2 = nn.AvgPool2D(kernel_size=2, stride=2, padding=0)
        self.pool4x4 = nn.AvgPool2D(kernel_size=4, stride=4, padding=0)
        self.flatten1x1 = nn.Flatten()
        self.flatten2x2 = nn.Flatten()
        self.flatten4x4 = nn.Flatten()

        # Fully connected layer between Block 1 and Block 2
        self.fc1 = nn.Linear(in_features=64 + 16 + 4, out_features=128)
        self.reshape = nn.ReshapeTransform(in_event_shape='mask', out_event_shape='mask')


        # Block 2: Feature Extraction Branches
        self.branch1 = nn.Sequential(
            nn.Conv2D(in_channels=128, out_channels=32, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2D(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )

        self.branch2 = nn.Sequential(
            nn.Conv2D(in_channels=128, out_channels=32, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2D(in_channels=32, out_channels=32, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2D(in_channels=32, out_channels=32, kernel_size=7, stride=1, padding=3),
            nn.ReLU(),
            nn.Conv2D(in_channels=32, out_channels=32, kernel_size=7, stride=1, padding=3),
            nn.ReLU()
        )

        self.branch3 = nn.AvgPool2D(kernel_size=2, stride=2, padding=0)

        # Fully connected layers for classification
        self.fc2 = nn.Linear(in_features=32 * 32 + 32, out_features=64)
        self.fc3 = nn.Linear(in_features=64, out_features=10)


    def forward(self, x):
        # Block 1
        pool1x1_out = self.pool1x1(x)
        pool2x2_out = self.pool2x2(x)
        pool4x4_out = self.pool4x4(x)

        flat1x1_out = self.flatten1x1(pool1x1_out)
        flat2x2_out = self.flatten2x2(pool2x2_out)
        flat4x4_out = self.flatten4x4(pool4x4_out)

        block1_out = paddle.concat([flat1x1_out, flat2x2_out, flat4x4_out], axis=1)

        # FC layer and reshape
        fc1_out = self.fc1(block1_out)
        reshaped_out = self.reshape(fc1_out)

        # Block 2
        branch1_out = self.branch1(reshaped_out)
        branch2_out = self.branch2(reshaped_out)
        branch3_out = self.branch3(reshaped_out)

        block2_out = paddle.concat([branch1_out, branch2_out, branch3_out], axis=1)
        flat_block2_out = paddle.flatten(block2_out, start_axis=1)

        # Classification
        fc2_out = self.fc2(flat_block2_out)
        fc2_out = F.relu(fc2_out)
        output = self.fc3(fc2_out)

        return output

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn([1, 1, 28, 28])  # Example input tensor
    output_tensor = model(input_tensor)
    print(output_tensor.shape)