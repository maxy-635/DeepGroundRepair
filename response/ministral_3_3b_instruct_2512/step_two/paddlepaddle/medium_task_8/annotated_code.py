import paddle
import paddle.nn as nn

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()
        # Main path
        self.main_path = nn.Sequential(
            nn.Conv2D(3, 64, kernel_size=3, stride=1, padding=1),
            nn.Conv2D(64, 32, kernel_size=3, stride=1, padding=1),
            nn.Conv2D(32, 128, kernel_size=3, stride=1, padding=1)
        )

        # Branch path
        self.branch_path = nn.Sequential(
            nn.Conv2D(3, 32, kernel_size=1, stride=1, padding=0)
        )

        # Fully connected layer
        self.fc = nn.Linear(128 * 8 * 8, 10)

    def forward(self, x):
        # the initial shape of x is: (1, 3, 32, 32)
        # Split the input into three groups along the last dimension
        group1 = x[:, :, :, :]  # output tensor shape: (1, 3, 32, 32)
        group2 = x[:, :, :, 1:]  # output tensor shape: (1, 3, 32, 31)
        group3 = x[:, :, :, 2:]  # output tensor shape: (1, 3, 32, 30)

        # Process group2 through the main path
        processed_group2 = self.main_path(group2)  # output tensor shape: (1, 128, 32, 31)

        # Combine group2 and group3
        combined_group23 = paddle.add(processed_group2, group3)

        # Process the combined result through the main path
        main_path_output = self.main_path(combined_group23)

        # Process the input through the branch path
        branch_path_output = self.branch_path(x)

        # Add outputs from main and branch paths
        fused_output = paddle.add(main_path_output, branch_path_output)

        # Flatten and pass through fully connected layer
        flattened_output = paddle.flatten(fused_output, start_dim=1)
        output = self.fc(flattened_output)

        return output

if __name__ == "__main__":
    # Create a dummy input tensor with shape (batch_size, channels, height, width)
    input_tensor = paddle.randn((1, 3, 32, 32))
    model = DL_Model()
    output_tensor = model(input_tensor)
    print("Input shape:", input_tensor.shape)
    print("Output shape:", output_tensor.shape)