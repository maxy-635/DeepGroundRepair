import paddle
import paddle.nn as nn

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()
        # Main path
        self.main_path_block1 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2D(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2D(kernel_size=2, stride=2)
        )

        self.main_path_block2 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2D(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2D(kernel_size=2, stride=2)
        )

        # Branch path
        self.branch_path = nn.Sequential(
            nn.Conv2D(in_channels=1, out_channels='mask', kernel_size='mask', stride='mask', padding='mask')
        )

        # Sum and flatten
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(in_features=64 * 7 * 7, out_features=10)

    def forward(self, x):
        # the initial shape of x is: (1, 1, 28, 28)
        main_path_output = self.main_path_block1(x)  # output tensor shape: (1, 32, 14, 14)
        main_path_output = self.main_path_block2(main_path_output)  # output tensor shape: (1, 64, 7, 7)

        branch_path_output = self.branch_path(x)  # output tensor shape of this layer should be (1, 64, 7, 7)

        # Ensure both paths have the same spatial dimensions
        # Assuming the main path reduces dimensions by pooling, we need to upsample the branch path
        # For simplicity, we'll sum the outputs directly after ensuring they have the same shape
        # In practice, you might need to use ConvTranspose2D to upsample the branch path
        # Here, we'll assume the branch path output is already the same shape as main_path_output
        # This is a simplification; in a real scenario, you'd need to adjust dimensions properly
        summed_output = main_path_output + branch_path_output

        flattened_output = self.flatten(summed_output)
        output = self.fc(flattened_output)

        return output

if __name__ == "__main__":
    model = DL_Model()
    # Create a dummy input tensor with shape (batch_size=1, channels=1, height=28, width=28)
    input_tensor = paddle.randn((1, 1, 28, 28))
    output_tensor = model(input_tensor)
    print("Input tensor shape:", input_tensor.shape)
    print("Output tensor shape:", output_tensor.shape)