import paddle
import paddle.nn as nn

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Main Path
        self.main_conv1 = nn.Conv2D(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.main_relu1 = nn.ReLU()
        self.main_conv2 = nn.Conv2D(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.main_relu2 = nn.ReLU()
        self.main_pool = nn.MaxPool2D(kernel_size=2, stride=2)

        # Branch Path
        self.branch_conv1 = nn.Conv2D(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.branch_relu1 = nn.ReLU()
        self.branch_pool = nn.MaxPool2D(kernel_size=2, stride=2)

        # Combination and Classification
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_features=64 * 8 * 8, out_features=128)
        self.relu3 = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(in_features=128, out_features=10)

    def forward(self, x):
        # the initial shape of x is: (1, 3, 32, 32)
        # Main Path
        main_path = self.main_conv1(x)  # output tensor shape: (1, 32, 32, 32)
        main_path = self.main_relu1(main_path)  # output tensor shape: (1, 32, 32, 32)
        main_path = self.main_conv2(main_path)  # output tensor shape: (1, 64, 32, 32)
        main_path = self.main_relu2(main_path)  # output tensor shape: (1, 64, 32, 32)
        main_path = self.main_pool(main_path)  # output tensor shape: (1, 64, 16, 16)

        # Branch Path
        branch_path = self.branch_conv1(x)  # output tensor shape of this layer should be (1, 64, 32, 32)
        branch_path = self.branch_relu1(branch_path)
        branch_path = self.branch_pool(branch_path)

        # Combination
        combined_output = main_path + branch_path

        # Classification
        combined_output = self.flatten(combined_output)
        combined_output = self.fc1(combined_output)
        combined_output = self.relu3(combined_output)
        combined_output = self.dropout(combined_output)
        combined_output = self.fc2(combined_output)

        return combined_output


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn(shape=[1, 3, 32, 32])
    output_tensor = model(input_tensor)
    print(output_tensor.shape)