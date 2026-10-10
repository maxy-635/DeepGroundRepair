import paddle
import paddle.nn as nn

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()
        # Main pathway: Split into three groups along the channel dimension
        self.main_pathway = nn.Sequential(
            nn.Conv2D(in_channels='mask', out_channels='mask', kernel_size='mask', stride='mask', padding='mask'),
            nn.Conv2D(64, 64, kernel_size=3, stride=1, padding=1),
            nn.Dropout(0.2),
            nn.Conv2D(64, 64, kernel_size=3, stride=1, padding=1),
            nn.Dropout(0.2),
            nn.Conv2D(64, 64, kernel_size=3, stride=1, padding=1),
            nn.Dropout(0.2)
        )

        # Branch pathway: 1x1 convolution to match main pathway output dimension
        self.branch_pathway = nn.Sequential(
            nn.Conv2D(3, 64, kernel_size=1, stride=1, padding=0)
        )

        # Layer normalization for the main pathway
        self.layer_norm_main = nn.LayerNorm(normalized_shape=[64])

        # Layer normalization for the branch pathway
        self.layer_norm_branch = nn.LayerNorm(normalized_shape=[64])

        # Fully connected layer for classification
        self.fc = nn.Linear(64 * 32 * 32, 10)

    def forward(self, x):
        # the initial shape of x is: (1, 3, 32, 32)
        # Split into three groups along the channel dimension
        x1, x2, x3 = paddle.split(x, 3, axis=1)  # output tensor shape: (1, 1, 32, 32); (1, 1, 32, 32); (1, 1, 32, 32)

        # Process each group through the main pathway
        out1 = self.main_pathway(x1)
        out2 = self.main_pathway(x2)
        out3 = self.main_pathway(x3)

        # Concatenate outputs from the three groups
        main_output = paddle.concat([out1, out2, out3], axis=1)

        # Normalize the main pathway output
        main_output = self.layer_norm_main(main_output)

        # Process the branch pathway
        branch_output = self.branch_pathway(x)
        branch_output = self.layer_norm_branch(branch_output)

        # Combine outputs from both pathways using addition
        combined_output = paddle.add(main_output, branch_output)

        # Flatten and pass through fully connected layer
        flattened_output = paddle.flatten(combined_output, start_axis=1)
        output = self.fc(flattened_output)

        return output

if __name__ == "__main__":
    # Create a dummy input tensor with shape (batch_size, channels, height, width)
    input_tensor = paddle.randn((1, 3, 32, 32))
    model = DL_Model()
    output_tensor = model(input_tensor)
    print("Input shape:", input_tensor.shape)
    print("Output shape:", output_tensor.shape)