import paddle
import paddle.nn as nn

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()
        in_channels = 3
        num_kernels = in_channels // 3
        
        # Split into three groups and apply 1x1 convolutions
        self.conv1_1 = nn.Conv2D(in_channels=in_channels // 3, out_channels=num_kernels, kernel_size=1, stride=1, padding=0)
        self.conv1_2 = nn.Conv2D(in_channels=in_channels // 3, out_channels=num_kernels, kernel_size=1, stride=1, padding=0)
        self.conv1_3 = nn.Conv2D(in_channels=in_channels // 3, out_channels=num_kernels, kernel_size=1, stride=1, padding=0)
        
        # Downsampling with average pooling
        self.pool1 = nn.AvgPool2D(kernel_size=2, stride=2, padding=0)
        
        # Concatenate the three groups
        self.concat = nn.Concat(axis='mask')
        
        # Flatten the concatenated feature maps
        self.flatten = nn.Flatten()
        
        # Fully connected layers for classification
        self.fc1 = nn.Linear(in_features=num_kernels * 3 * 16 * 16, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=10)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # Split the input into three groups
        x1 = x[:, :1, :, :]
        x2 = x[:, 1:2, :, :]
        x3 = x[:, 2:, :, :]
        
        # Apply 1x1 convolutions to each group
        x1 = self.conv1_1(x1)
        x2 = self.conv1_2(x2)
        x3 = self.conv1_3(x3)
        
        # Downsample each group
        x1 = self.pool1(x1)
        x2 = self.pool1(x2)
        x3 = self.pool1(x3)
        
        # Concatenate the three groups
        x = self.concat([x1, x2, x3])
        
        # Flatten the concatenated feature maps
        x = self.flatten(x)
        
        # Pass through fully connected layers
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        
        return x

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn([1, 3, 32, 32])
    output_tensor = model(input_tensor)
    print(output_tensor.shape)