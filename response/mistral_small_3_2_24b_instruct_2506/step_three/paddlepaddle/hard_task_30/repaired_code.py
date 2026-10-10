import paddle
import paddle.nn as nn
import paddle.nn.functional as F

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()

        # First block: Dual-path structure
        self.conv1 = nn.Conv2D(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2D(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2D(in_channels=128, out_channels=3, kernel_size=3, stride=1, padding=1)

        # Second block: Multi-group feature extraction
        self.depthwise_conv1 = nn.Conv2D(in_channels=1, out_channels=1, kernel_size=1, stride=1, padding=0, groups=1)
        self.depthwise_conv3 = nn.Conv2D(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=1, groups=1)
        self.depthwise_conv5 = nn.Conv2D(in_channels=1, out_channels=1, kernel_size=5, stride=1, padding=2, groups=1)

        # Fully connected layers
        self.fc1 = nn.Linear(in_features=32*32*3, out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=10)

    def forward(self, x):
        # First block: Dual-path structure
        main_path = self.conv1(x)
        main_path = self.conv2(main_path)
        main_path = self.conv3(main_path)
        output_block1 = paddle.add(main_path, x)

        # Second block: Multi-group feature extraction
        x_split = paddle.split(output_block1, num_or_sections=3, axis=1)
        group1 = self.depthwise_conv1(x_split[0])
        group2 = self.depthwise_conv3(x_split[1])
        group3 = self.depthwise_conv5(x_split[2])
        output_block2 = paddle.concat([group1, group2, group3], axis=1)

        # Fully connected layers
        output_block2_flat = paddle.flatten(output_block2, start_axis=1)
        fc_output = self.fc1(output_block2_flat)
        fc_output = F.relu(fc_output)
        final_output = self.fc2(fc_output)

        return final_output

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn(shape=[1, 3, 32, 32])
    output_tensor = model(input_tensor)
    print(output_tensor.shape)