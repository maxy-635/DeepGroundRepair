import paddle
import paddle.nn as nn

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Block 1
        self.conv1_1 = nn.Conv2D(in_channels=3, out_channels=16, kernel_size=1, stride=1, padding=0)
        self.conv1_2 = nn.Conv2D(in_channels=16, out_channels='mask', kernel_size=3, stride=1, padding=1)
        self.conv1_3 = nn.Conv2D(in_channels=32, out_channels=16, kernel_size=1, stride=1, padding=0)
        self.group_norm1 = nn.GroupNorm(num_groups=1, num_channels=16)
        self.relu1 = nn.ReLU()

        # Transition Convolution
        self.transition_conv = nn.Conv2D(in_channels=16, out_channels=64, kernel_size=1, stride=1, padding=0)
        self.transition_group_norm = nn.GroupNorm(num_groups=1, num_channels=64)
        self.transition_relu = nn.ReLU()


        # Block 2
        self.adaptive_avg_pool = nn.AdaptiveAvgPool2D(output_size=1)
        self.fc1 = nn.Linear(in_features=64, out_features=128)
        self.relu2 = nn.ReLU()
        self.fc2 = nn.Linear(in_features=128, out_features=64)
        self.fc2_reshape = None  # Will be populated in the forward pass

        # Branch
        self.branch_conv = nn.Conv2D(in_channels=3, out_channels=64, kernel_size=1, stride=1, padding=0)
        self.branch_relu = nn.ReLU()

        # Final Classification
        self.final_fc = nn.Linear(in_features=64, out_features=10)

    def forward(self, x):
        # the initial shape of x is: (1, 3, 32, 32)
        # Block 1
        x1_1 = self.conv1_1(x)  # output tensor shape: (1, 16, 32, 32)
        x1_2 = self.conv1_2(x1_1)  # output tensor shape of this layer should be (1, 16, 32, 32)
        x1_3 = self.conv1_3(x1_2)  # output tensor shape: (1, 16, 32, 32)
        x1 = paddle.concat([x1_1, x1_2, x1_3], axis=-1)
        x1 = self.group_norm1(x1)
        x1 = self.relu1(x1)

        # Transition Convolution
        x_transition = self.transition_conv(x1)
        x_transition = self.transition_group_norm(x_transition)
        x_transition = self.transition_relu(x_transition)

        # Block 2
        x_adaptive_pool = self.adaptive_avg_pool(x_transition)
        x_flatten = paddle.flatten(x_adaptive_pool, start_axis=1)
        x_fc1 = self.fc1(x_flatten)
        x_relu2 = self.relu2(x_fc1)
        self.fc2_reshape = self.fc2(x_relu2)
        self.fc2_reshape = self.fc2_reshape.reshape(-1, 64, 1, 1)

        x_main_path = x_transition * self.fc2_reshape

        # Branch
        x_branch = self.branch_conv(x)
        x_branch = self.branch_relu(x_branch)

        # Add and Classify
        x_added = x_main_path + x_branch
        x_flattened = paddle.flatten(x_added, start_axis=1)
        x_output = self.final_fc(x_flattened)

        return x_output


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn(shape=[1, 3, 32, 32])  # Dummy input: batch_size=1, channels=3, height=32, width=32
    output_tensor = model(input_tensor)
    print(output_tensor.shape)