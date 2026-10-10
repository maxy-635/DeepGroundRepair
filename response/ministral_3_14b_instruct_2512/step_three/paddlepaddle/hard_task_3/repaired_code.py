import paddle
import paddle.nn as nn
import paddle.nn.functional as F

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()
        # Main pathway
        self.conv1x1_1 = nn.Conv2D(in_channels=1, out_channels=64, kernel_size=1, stride=1, padding=0)
        self.conv3x3_1 = nn.Conv2D(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.dropout_1 = nn.Dropout(p=0.5)

        self.conv1x1_2 = nn.Conv2D(in_channels=3, out_channels=64, kernel_size=1, stride=1, padding=0)
        self.conv3x3_2 = nn.Conv2D(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.dropout_2 = nn.Dropout(p=0.5)

        self.conv1x1_3 = nn.Conv2D(in_channels=3, out_channels=64, kernel_size=1, stride=1, padding=0)
        self.conv3x3_3 = nn.Conv2D(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.dropout_3 = nn.Dropout(p=0.5)

        # Branch pathway
        self.conv1x1_branch = nn.Conv2D(in_channels=3, out_channels=192, kernel_size=1, stride=1, padding=0)

        # Concatenation and addition
        self.concat = paddle.concat

        # Final fully connected layer
        self.fc = nn.Linear(in_features=192*32*32, out_features=10)  # Assuming CIFAR-10 output size

    def forward(self, x):
        # the initial shape of x is: (1, 3, 32, 32)
        # Split input into three groups along channel dimension
        x1, x2, x3 = paddle.split(x, num_or_sections=3, axis=1)  # output tensor shape: (1, 1, 32, 32); (1, 1, 32, 32); (1, 1, 32, 32)

        # Main pathway processing
        x1 = self.conv1x1_1(x1)
        x1 = self.conv3x3_1(x1)
        x1 = self.dropout_1(x1)

        x2 = self.conv1x1_2(x2)
        x2 = self.conv3x3_2(x2)
        x2 = self.dropout_2(x2)

        x3 = self.conv1x1_3(x3)
        x3 = self.conv3x3_3(x3)
        x3 = self.dropout_3(x3)

        # Concatenate the three groups
        main_path = self.concat([x1, x2, x3], axis=1)

        # Branch pathway processing
        branch_path = self.conv1x1_branch(x)
        branch_path = F.avg_pool2d(branch_path, kernel_size=3, stride=3, padding=1)

        # Combine main and branch pathways
        combined = main_path + branch_path

        # Flatten and pass through fully connected layer
        combined_flat = paddle.flatten(combined, start_axis=1)
        output = self.fc(combined_flat)

        return output

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn((1, 3, 32, 32))  # Batch size 1, 3 channels, 32x32 image
    output_tensor = model(input_tensor)
    print(output_tensor.shape)