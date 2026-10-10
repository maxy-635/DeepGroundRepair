import paddle
import paddle.nn as nn
import paddle.nn.functional as F

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Block 1: Main and Branch Paths
        self.main_path = nn.Sequential(
            nn.Conv2D(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Dropout(p=0.25)
        )
        self.restore_channels = nn.Conv2D(in_channels=16, out_channels=3, kernel_size=1, stride=1, padding=0)
        self.branch_path = nn.Identity()  # Direct connection

        # Block 2: Separable Convolution Groups
        self.sep_conv1 = nn.Sequential(
            nn.Conv2D(in_channels=3, out_channels=8, kernel_size=1, stride=1, padding=0, groups=1),
            nn.ReLU(),
            nn.Dropout(p=0.25)
        )
        self.sep_conv3 = nn.Sequential(
            nn.Conv2D(in_channels=3, out_channels=8, kernel_size=3, stride=1, padding=1, groups=1),
            nn.ReLU(),
            nn.Dropout(p=0.25)
        )
        self.sep_conv5 = nn.Sequential(
            nn.Conv2D(in_channels=3, out_channels=8, kernel_size=5, stride=1, padding=2, groups=1),
            nn.ReLU(),
            nn.Dropout(p=0.25)
        )
        
        # Final Layers
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(in_features=8 * 32 * 32, out_features=10)  # Assuming input size 32x32

    def forward(self, x):
        # Block 1
        main_path_output = self.main_path(x)
        restore_output = self.restore_channels(main_path_output)
        branch_path_output = self.branch_path(x)
        block1_output = restore_output + branch_path_output
        
        # Block 2
        # Split input along the last dimension
        x1 = block1_output[:, :, :, :1]
        x3 = block1_output[:, :, :, 1:2]
        x5 = block1_output[:, :, :, 2:]
        
        sep_conv1_output = self.sep_conv1(x1)
        sep_conv3_output = self.sep_conv3(x3)
        sep_conv5_output = self.sep_conv5(x5)
        
        # Concatenate outputs
        concat_output = paddle.concat([sep_conv1_output, sep_conv3_output, sep_conv5_output], axis=3)

        # Final Layers
        flattened_output = self.flatten(concat_output)
        output = self.linear(flattened_output)
        return output


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn([1, 3, 32, 32])
    output_tensor = model(input_tensor)
    print(output_tensor.shape)