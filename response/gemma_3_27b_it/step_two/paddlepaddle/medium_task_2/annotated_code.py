import paddle
import paddle.nn as nn

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()
        
        # Main path
        self.main_path_conv1 = nn.Conv2D(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.main_path_relu1 = nn.ReLU()
        self.main_path_conv2 = nn.Conv2D(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.main_path_relu2 = nn.ReLU()
        self.main_path_pool = nn.MaxPool2D(kernel_size=2, stride=2, padding=0)
        
        # Branch path
        self.branch_path_conv = nn.Conv2D(in_channels=3, out_channels=32, kernel_size='mask', stride='mask', padding='mask')
        self.branch_path_relu = nn.ReLU()
        
        # Combined layers
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_features=64 * 8 * 8 + 32 * 8 * 8, out_features=128)  # Assuming input size 32x32
        self.fc1_relu = nn.ReLU()
        self.fc2 = nn.Linear(in_features=128, out_features=10)
        self.softmax = nn.Softmax(axis=1)

    def forward(self, x):
        # the initial shape of x is: (1, 3, 32, 32)
        # Main path
        main_path = self.main_path_conv1(x)  # output tensor shape: (1, 32, 32, 32)
        main_path = self.main_path_relu1(main_path)  # output tensor shape: (1, 32, 32, 32)
        main_path = self.main_path_conv2(main_path)  # output tensor shape: (1, 64, 32, 32)
        main_path = self.main_path_relu2(main_path)  # output tensor shape: (1, 64, 32, 32)
        main_path = self.main_path_pool(main_path)  # output tensor shape: (1, 64, 16, 16)
        
        # Branch path
        branch_path = self.branch_path_conv(x)  # output tensor shape of this layer should be (1, 32, 16, 16)
        branch_path = self.branch_path_relu(branch_path)
        
        # Concatenate features
        combined_features = paddle.concat([main_path, branch_path], axis=1)
        
        # Flatten and fully connected layers
        flattened = self.flatten(combined_features)
        fc1_output = self.fc1(flattened)
        fc1_output = self.fc1_relu(fc1_output)
        fc2_output = self.fc2(fc1_output)
        probabilities = self.softmax(fc2_output)
        
        return probabilities


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn([1, 3, 32, 32])  # Batch size of 1, 3 channels, 32x32 image
    output_tensor = model(input_tensor)
    print(output_tensor.shape) # Expected output shape: [1, 10]