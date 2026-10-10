import paddle
import paddle.nn as nn
import paddle.nn.functional as F

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Main Pathway
        self.conv1_group1 = nn.Conv2D(in_channels=3, out_channels=32, kernel_size=1, stride=1, padding=0)
        self.conv2_group1 = nn.Conv2D(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.dropout1 = nn.Dropout(p=0.5)

        self.conv1_group2 = nn.Conv2D(in_channels=3, out_channels=32, kernel_size=1, stride=1, padding=0)
        self.conv2_group2 = nn.Conv2D(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.dropout2 = nn.Dropout(p=0.5)

        self.conv1_group3 = nn.Conv2D(in_channels=3, out_channels=32, kernel_size=1, stride=1, padding=0)
        self.conv2_group3 = nn.Conv2D(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.dropout3 = nn.Dropout(p=0.5)

        # Branch Pathway
        self.conv_branch = nn.Conv2D(in_channels=3, out_channels=192, kernel_size=1, stride=1, padding=0)

        # Fully Connected Layer
        self.fc = nn.Linear(in_features=192 * 32 * 32, out_features=10)  # Assuming input size is 32x32

    def forward(self, x):
        # Main Pathway
        group1 = F.relu(self.conv1_group1(x))
        group1 = F.relu(self.conv2_group1(group1))
        group1 = self.dropout1(group1)

        group2 = F.relu(self.conv1_group2(x))
        group2 = F.relu(self.conv2_group2(group2))
        group2 = self.dropout2(group2)

        group3 = F.relu(self.conv1_group3(x))
        group3 = F.relu(self.conv2_group3(group3))
        group3 = self.dropout3(group3)

        main_path = paddle.concat([group1, group2, group3], axis=1)

        # Branch Pathway
        branch_path = self.conv_branch(x)

        # Combine Pathways
        output = paddle.add(main_path, branch_path)

        # Flatten and Fully Connected Layer
        output = output.reshape([output.shape[0], -1])
        output = self.fc(output)

        return output


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn([1, 3, 32, 32])  # Batch size of 1, 3 channels, 32x32 image
    output_tensor = model(input_tensor)
    print(output_tensor.shape)