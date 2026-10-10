import paddle
import paddle.nn as nn
import paddle.nn.functional as F

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Block 1: Main and Branch Paths
        self.main_path_conv1 = nn.Conv2D(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.main_path_conv2 = nn.Conv2D(in_channels=32, out_channels=1, kernel_size=3, stride=1, padding=1)
        self.branch_path = nn.Identity()  # Direct connection to input

        # Block 2: Max Pooling Layers
        self.pool1 = nn.MaxPool2D(kernel_size=1, stride=1, padding=0)
        self.pool2 = nn.MaxPool2D(kernel_size=2, stride=2, padding=0)
        self.pool3 = nn.MaxPool2D(kernel_size=4, stride=4, padding=0)
        self.flatten1 = nn.Flatten()
        self.flatten2 = nn.Flatten()
        self.flatten3 = nn.Flatten()

        # Fully Connected Layers
        self.fc1 = nn.Linear(in_features=1600, out_features=128)  # Adjusted input size
        self.fc2 = nn.Linear(in_features=128, out_features=10)
        self.relu = nn.ReLU()


    def forward(self, x):
        # Block 1
        main_path_output = self.main_path_conv1(x)
        main_path_output = self.main_path_conv2(main_path_output)
        branch_path_output = self.branch_path(x)
        block1_output = main_path_output + branch_path_output

        # Block 2
        pool1_output = self.pool1(block1_output)
        pool2_output = self.pool2(block1_output)
        pool3_output = self.pool3(block1_output)

        flatten1_output = self.flatten1(pool1_output)
        flatten2_output = self.flatten2(pool2_output)
        flatten3_output = self.flatten3(pool3_output)

        concat_output = paddle.concat([flatten1_output, flatten2_output, flatten3_output], axis=1)

        # Fully Connected Layers
        fc1_output = self.fc1(concat_output)
        fc1_output = self.relu(fc1_output)
        output = self.fc2(fc1_output)

        return output


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn([1, 1, 28, 28])
    output_tensor = model(input_tensor)
    print(output_tensor.shape)