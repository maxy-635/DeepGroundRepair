import paddle
import paddle.nn as nn
import paddle.nn.functional as F

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Feature extraction path 1: 1x1 convolution
        self.conv1x1 = nn.Conv2D(in_channels=3, out_channels=64, kernel_size=1, stride=1, padding=0)
        self.relu1 = nn.ReLU()
        self.layer_norm1 = nn.LayerNorm(normalized_shape=[64, 32, 32])

        # Feature extraction path 2: 1x1, 1x7, 7x1 convolutions
        self.conv1x1_path2 = nn.Conv2D(in_channels=3, out_channels=64, kernel_size=1, stride=1, padding=0)
        self.relu2_1 = nn.ReLU()
        self.conv1x7_path2 = nn.Conv2D(in_channels=64, out_channels=64, kernel_size=(1, 7), stride=1, padding=0)
        self.relu2_2 = nn.ReLU()
        self.conv7x1_path2 = nn.Conv2D(in_channels=64, out_channels=64, kernel_size=(7, 1), stride=1, padding=0)
        self.relu2_3 = nn.ReLU()
        self.layer_norm2 = nn.LayerNorm(normalized_shape='mask')

        # Concatenation and alignment
        self.concat = paddle.concat
        self.conv1x1_align = nn.Conv2D(in_channels=128, out_channels=3, kernel_size=1, stride=1, padding=0)
        self.relu_align = nn.ReLU()

        # Branch to input
        self.add = paddle.add

        # Fully connected layers for classification
        self.fc1 = nn.Linear(in_features=3 * 32 * 32, out_features=128)
        self.relu_fc1 = nn.ReLU()
        self.fc2 = nn.Linear(in_features=128, out_features=10)

    def forward(self, x):
        # the initial shape of x is: (1, 3, 32, 32)
        # Path 1
        path1 = self.conv1x1(x)  # output tensor shape: (1, 64, 32, 32)
        path1 = self.relu1(path1)  # output tensor shape: (1, 64, 32, 32)
        path1 = self.layer_norm1(path1)  # output tensor shape: (1, 64, 32, 32)

        # Path 2
        path2 = self.conv1x1_path2(x)  # output tensor shape: (1, 64, 32, 32)
        path2 = self.relu2_1(path2)  # output tensor shape: (1, 64, 32, 32)
        path2 = self.conv1x7_path2(path2)  # output tensor shape: (1, 64, 32, 26)
        path2 = self.relu2_2(path2)  # output tensor shape: (1, 64, 32, 26)
        path2 = self.conv7x1_path2(path2)  # output tensor shape: (1, 64, 26, 26)
        path2 = self.relu2_3(path2)  # output tensor shape: (1, 64, 26, 26)
        path2 = self.layer_norm2(path2)
        
        # Concatenate and align
        concat_output = self.concat([path1, path2], axis=1)
        aligned_output = self.conv1x1_align(concat_output)
        aligned_output = self.relu_align(aligned_output)

        # Add branch to input
        output = self.add(x, aligned_output)
        
        # Flatten for fully connected layers
        output = paddle.reshape(output, [output.shape[0], -1])

        # Fully connected layers
        output = self.fc1(output)
        output = self.relu_fc1(output)
        output = self.fc2(output)

        return output


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn([1, 3, 32, 32])  # Batch size of 1, 3 channels, 32x32 image
    output_tensor = model(input_tensor)
    print(output_tensor.shape)  # Expected output shape: [1, 10]