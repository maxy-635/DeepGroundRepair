import paddle
import paddle.nn as nn
import paddle.nn.functional as F

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()
        
        # Main Path
        self.main_relu1 = nn.ReLU()
        self.main_conv1 = nn.Conv2D(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1, groups=1, bias_attr=True)
        self.main_relu2 = nn.ReLU()
        self.main_conv2 = nn.Conv2D(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1, groups=1, bias_attr=True)
        self.main_pool = nn.MaxPool2D(kernel_size=2, stride=2, padding=0)

        # Branch Path
        self.branch_conv = nn.Conv2D(in_channels=1, out_channels=64, kernel_size=1, stride=1, padding=0, groups=1, bias_attr=True)

        # Final Layers
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(in_features=64 * 7 * 7, out_features=10)  # Adjust input features based on MNIST size (28x28 -> 7x7 after two 2x2 max pools)

    def forward(self, x):
        # Main Path
        main_path = self.main_relu1(self.main_conv1(x))
        main_path = self.main_relu2(self.main_conv2(main_path))
        main_path = self.main_pool(main_path)

        # Branch Path
        branch_path = self.branch_conv(x)

        # Sum Paths
        merged_path = main_path + branch_path

        # Final Layers
        flattened_path = self.flatten(merged_path)
        output = self.linear(flattened_path)
        return output


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn([1, 1, 28, 28])  # Batch size 1, 1 channel, 28x28 image
    output_tensor = model(input_tensor)
    print(output_tensor.shape)  # Expected output: paddle.Tensor, shape=[1, 10]