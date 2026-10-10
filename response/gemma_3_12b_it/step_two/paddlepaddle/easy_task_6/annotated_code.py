import paddle
import paddle.nn as nn

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Main path
        self.conv1 = nn.Conv2D(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2D(in_channels=16, out_channels=1, kernel_size=3, stride=1, padding=1)

        # Branch path
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(in_features=784, out_features=10)

    def forward(self, x):
        # the initial shape of x is: (1, 1, 28, 28)
        # Main path
        main_path = self.conv1(x)  # output tensor shape: (1, 16, 28, 28)
        main_path = self.relu1(main_path)  # output tensor shape: (1, 16, 28, 28)
        main_path = self.conv2(main_path)  # output tensor shape: (1, 1, 28, 28)

        # Branch path
        branch_path = self.flatten(x)  # output tensor shape: (1, 784)
        
        # Combine the paths
        combined = paddle.add(main_path, branch_path)
        
        # Flatten and fully connected layer
        flattened = self.flatten(combined)
        output = self.linear(flattened)
        return output


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.rand(shape=(1, 1, 28, 28))
    output_tensor = model(input_tensor)
    print(output_tensor.shape)