import paddle
import paddle.nn as nn

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Split the input into three groups along the channel dimension
        self.group1 = nn.Conv2D(in_channels=3, out_channels=32, kernel_size=1, stride=1, padding=0)
        self.group2 = nn.Conv2D(in_channels=3, out_channels=32, kernel_size=1, stride=1, padding=0)
        self.group3 = nn.Conv2D(in_channels=3, out_channels=32, kernel_size=1, stride=1, padding=0)

        self.conv1_1 = nn.Conv2D(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv1_2 = nn.Conv2D(in_channels=64, out_channels=32, kernel_size=1, stride=1, padding=0)

        self.conv2_1 = nn.Conv2D(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv2_2 = nn.Conv2D(in_channels=64, out_channels=32, kernel_size=1, stride=1, padding=0)

        self.conv3_1 = nn.Conv2D(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv3_2 = nn.Conv2D(in_channels=64, out_channels='mask', kernel_size=1, stride=1, padding=0)
        
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(in_features=32 * 32 * 32, out_features=10)

    def forward(self, x):
        # the initial shape of x is: (1, 3, 32, 32)
        # Process each group
        group1 = self.group1(x)  # output tensor shape: (1, 32, 32, 32)
        group1 = self.conv1_1(group1)  # output tensor shape: (1, 64, 32, 32)
        group1 = self.conv1_2(group1)  # output tensor shape: (1, 32, 32, 32)

        group2 = self.group2(x)  # output tensor shape: (1, 32, 32, 32)
        group2 = self.conv2_1(group2)  # output tensor shape: (1, 64, 32, 32)
        group2 = self.conv2_2(group2)  # output tensor shape: (1, 32, 32, 32)

        group3 = self.group3(x)  # output tensor shape: (1, 32, 32, 32)
        group3 = self.conv3_1(group3)  # output tensor shape: (1, 64, 32, 32)
        group3 = self.conv3_2(group3)  # output tensor shape of this layer should be (1, 3, 32, 32)

        # Combine the outputs of the three groups
        main_path = paddle.add(group1, group2)
        main_path = paddle.add(main_path, group3)

        # Fuse with the original input
        combined_features = paddle.add(main_path, x)

        # Flatten and apply fully connected layer
        flattened_features = self.flatten(combined_features)
        output = self.linear(flattened_features)

        return output


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn(shape=(1, 3, 32, 32))
    output_tensor = model(input_tensor)
    print(output_tensor.shape)