import paddle
import paddle.nn as nn

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Block 1
        self.conv1_1 = nn.Conv2D(in_channels=3, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.conv1_2 = nn.Conv2D(in_channels=3, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.conv1_3 = nn.Conv2D(in_channels=3, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.concat1 = nn.Concat(dim='mask')

        # Block 2
        self.shape = paddle.Shape(self.conv1_1(paddle.randn(shape=(1,3,32,32))).shape)
        self.reshape2_1 = nn.Slice(input=self.conv1_1(paddle.randn(shape=(1,3,32,32))), axes=[0, 1, 2, 3], starts=[0, 0, 0, 0], ends=[self.shape[0], self.shape[1], self.shape[2], self.shape[3]])
        self.reshape2_2 = nn.Slice(input=self.reshape2_1, axes=[0, 1, 2, 3], starts=[0, 0, 0, 0], ends=[3, 3, 3, 3])
        self.perm2_1 = nn.Transpose(input=self.reshape2_2, perm=[0, 1, 3, 2])
        self.reshape2_3 = nn.Identity()
        self.reshape2_3 = nn.Slice(input=self.perm2_1, axes=[0, 1, 3, 2], starts=[0, 0, 0, 0], ends=[3, 3, 3, 3])
        self.shuffle2 = nn.Identity()
        self.shuffle2 = nn.Slice(input=self.reshape2_3, axes=[0, 1, 3, 2], starts=[0, 0, 0, 0], ends=[3, 3, 3, 3])


        # Block 3
        self.conv3_1 = nn.Conv2D(in_channels=3, out_channels=1, kernel_size=3, stride=1, padding=1)

        # Branch
        self.branch_input = paddle.randn(shape=(1,3,32,32))

        # Combination
        self.add_1 = nn.Add(input1=self.conv3_1(self.shuffle2(paddle.randn(shape=(1,3,32,32)))), input2=self.branch_input)

        # Fully Connected Layer
        self.fc = nn.fc(x=self.add_1, size=10, activation=nn.Sigmoid())

    def forward(self, x):
        # Block 1
        x1_1 = self.conv1_1(x)
        x1_2 = self.conv1_2(x)
        x1_3 = self.conv1_3(x)
        fused_features = self.concat1([x1_1, x1_2, x1_3])

        # Block 2
        x2_1 = self.shuffle2(paddle.randn(shape=(1,3,32,32)))
        # Block 3
        x3_1 = self.conv3_1(x2_1)

        # Branch
        branch_output = self.branch_input

        # Combination
        combined_output = self.add_1(x3_1, branch_output)

        # Fully Connected Layer
        output = self.fc(combined_output)

        return output

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn(shape=(1, 3, 32, 32))
    output_tensor = model(input_tensor)
    print(output_tensor.shape)