import paddle
import paddle.nn as nn

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()
        
        # Main pathway
        self.conv1x1_main = nn.Conv2D(in_channels=3, out_channels=64, kernel_size=1, stride=1, padding=0)
        self.relu_main = nn.ReLU()

        # Parallel branch
        self.conv1x1_branch = nn.Conv2D(in_channels=3, out_channels=64, kernel_size=1, stride=1, padding=0)
        self.conv1x3_branch = nn.Conv2D(in_channels=3, out_channels=64, kernel_size=(1, 3), stride=1, padding=(0, 1))
        self.conv3x1_branch = nn.Conv2D(in_channels=3, out_channels=64, kernel_size=(3, 1), stride=1, padding=(1, 0))
        self.relu_branch = nn.ReLU()

        # Concatenation and final convolution
        self.concat = nn.Concat(axis='mask')
        self.conv1x1_final = nn.Conv2D(in_channels=192, out_channels=3, kernel_size=1, stride=1, padding=0)

        # Direct connection
        self.identity = nn.Identity()

        # Fully connected layers
        self.fc1 = nn.Linear(in_features=3072, out_features=128)
        self.relu_fc1 = nn.ReLU()
        self.fc2 = nn.Linear(in_features=128, out_features=10)
        self.softmax = nn.Softmax(axis=1)

    def forward(self, x):
        # Main pathway
        main_path = self.conv1x1_main(x)
        main_path = self.relu_main(main_path)

        # Parallel branch
        branch1x1 = self.conv1x1_branch(x)
        branch1x1 = self.relu_branch(branch1x1)
        branch1x3 = self.conv1x3_branch(x)
        branch1x3 = self.relu_branch(branch1x3)
        branch3x1 = self.conv3x1_branch(x)
        branch3x1 = self.relu_branch(branch3x1)
        branch_path = self.concat([branch1x1, branch1x3, branch3x1])
        
        # Concatenation and final convolution
        combined_path = self.concat([main_path, branch_path], axis=1)
        combined_path = self.conv1x1_final(combined_path)

        # Direct connection
        residual = self.identity(x)
        output = paddle.add(combined_path, residual)

        # Flatten and fully connected layers
        output = paddle.reshape(output, [output.shape[0], -1])
        output = self.fc1(output)
        output = self.relu_fc1(output)
        output = self.fc2(output)
        output = self.softmax(output)
        
        return output

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn([1, 3, 32, 32])
    output_tensor = model(input_tensor)
    print(output_tensor.shape)