import paddle
import paddle.nn as nn

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()
        self.conv1 = nn.Conv2D(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2D(in_channels=32, out_channels=1, kernel_size=3, stride=1, padding=1)
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(in_features=28*28, out_features=10)

    def forward(self, x):
        main_path = self.conv1(x)
        main_path = self.relu1(main_path)
        main_path = self.conv2(main_path)
        
        branch_path = x
        
        combined_path = paddle.add(main_path, branch_path)
        
        flattened_path = self.flatten(combined_path)
        
        output = self.linear(flattened_path)
        return output


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn([1, 1, 28, 28])
    output_tensor = model(input_tensor)
    print(output_tensor.shape)