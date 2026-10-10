import paddle
import paddle.nn as nn
import paddle.nn.functional as F

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()

        # First block
        self.pool1 = nn.AvgPool2D(kernel_size=1, stride=1, padding=0)
        self.pool2 = nn.AvgPool2D(kernel_size=2, stride=2, padding=0)
        self.pool3 = nn.AvgPool2D(kernel_size=4, stride=4, padding=0)
        self.flatten1 = nn.Flatten()
        self.flatten2 = nn.Flatten()
        self.flatten3 = nn.Flatten()
        
        # Fully connected layer and reshape
        self.fc1 = nn.Linear(in_features=1029, out_features=128)
        
        # Second block
        self.conv1 = nn.Conv2D(in_channels=1, out_channels=1, kernel_size=1, stride=1, padding=0, groups=1)
        self.conv2 = nn.Conv2D(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=1, groups=1)
        self.conv3 = nn.Conv2D(in_channels=1, out_channels=1, kernel_size=5, stride=1, padding=2, groups=1)
        self.conv4 = nn.Conv2D(in_channels=1, out_channels=1, kernel_size=7, stride=1, padding=3, groups=1)
        self.flatten4 = nn.Flatten()
        self.fc2 = nn.Linear(in_features=128, out_features=10)

    def forward(self, x):
        # the initial shape of x is: (1, 1, 28, 28)
        # First block
        pool1_out = self.pool1(x)  # output tensor shape: (1, 1, 28, 28)
        pool2_out = self.pool2(x)  # output tensor shape: (1, 1, 14, 14)
        pool3_out = self.pool3(x)  # output tensor shape: (1, 1, 7, 7)
        
        flat1_out = self.flatten1(pool1_out)  # output tensor shape: (1, 784)
        flat2_out = self.flatten2(pool2_out)  # output tensor shape: (1, 196)
        flat3_out = self.flatten3(pool3_out)  # output tensor shape: (1, 49)
        
        concat_out = paddle.concat([flat1_out, flat2_out, flat3_out], axis=1)  # output tensor shape: (1, 1029)
        
        # Fully connected layer and reshape
        fc_out = self.fc1(concat_out)
        
        # Reshape for the second block
        fc_out = fc_out.reshape([fc_out.shape[0], 1, 8, 8])

        # Second block
        group1 = self.conv1(fc_out)
        group2 = self.conv2(fc_out)
        group3 = self.conv3(fc_out)
        group4 = self.conv4(fc_out)
        
        concat_groups = paddle.concat([group1, group2, group3, group4], axis=1)
        
        flat_out = self.flatten4(concat_groups)
        
        # Final fully connected layer
        output = self.fc2(flat_out)
        
        return output

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn([1, 1, 28, 28])
    output_tensor = model(input_tensor)
    print(output_tensor.shape)