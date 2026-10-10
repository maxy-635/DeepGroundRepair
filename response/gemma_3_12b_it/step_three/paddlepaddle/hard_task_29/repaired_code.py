import paddle
import paddle.nn as nn
import paddle.nn.functional as F

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Block 1: Main and Branch Paths
        self.conv1_main = nn.Conv2D(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2_main = nn.Conv2D(in_channels=16, out_channels=1, kernel_size=3, stride=1, padding=1)
        self.branch = nn.Conv2D(in_channels=1, out_channels=1, kernel_size=1, stride=1, padding=0) # Connect directly to input

        # Block 2: Max Pooling Layers
        self.pool1 = nn.MaxPool2D(kernel_size=1, stride=1, padding=0)
        self.pool2 = nn.MaxPool2D(kernel_size=2, stride=2, padding=0)
        self.pool3 = nn.MaxPool2D(kernel_size=4, stride=4, padding=0)

        # Flatten and Concatenate
        self.flatten1 = nn.Flatten()
        self.flatten2 = nn.Flatten()
        self.flatten3 = nn.Flatten()
        self.concat = nn.Concat([flatten1_output, flatten2_output, flatten3_output])

        # Fully Connected Layers
        self.fc1 = nn.Linear(in_features=16 + 1 + 16 + 1 + 16 + 1, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=10)

    def forward(self, x):
        # Block 1
        main_path = F.relu(self.conv1_main(x))
        main_path = F.relu(self.conv2_main(main_path))
        branch_path = self.branch(x)
        combined_path = main_path + branch_path

        # Block 2
        pool1_output = self.pool1(combined_path)
        pool2_output = self.pool2(combined_path)
        pool3_output = self.pool3(combined_path)

        flatten1_output = self.flatten1(pool1_output)
        flatten2_output = self.flatten2(pool2_output)
        flatten3_output = self.flatten3(pool3_output)

        concatenated_output = self.concat((flatten1_output, flatten2_output, flatten3_output))

        # Fully Connected Layers
        fc1_output = F.relu(self.fc1(concatenated_output))
        output = self.fc2(fc1_output)

        return output


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn(shape=[1, 1, 28, 28])  # Dummy input tensor (batch_size=1, channels=1, height=28, width=28)
    output_tensor = model(input_tensor)
    print(output_tensor.shape)