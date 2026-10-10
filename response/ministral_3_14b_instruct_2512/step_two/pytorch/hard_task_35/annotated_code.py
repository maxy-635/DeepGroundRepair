import torch
import torch.nn as nn
import torch.nn.functional as F

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Define the block that applies global average pooling followed by two fully connected layers
        self.block = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),  # Global average pooling
            nn.Flatten(),
            nn.Linear(in_features=512, out_features=256),  # Assuming input channels are 512
            nn.ReLU(inplace=True),
            nn.Linear(in_features=256, out_features=512),  # Output channels match input channels
            nn.Sigmoid()  # To get weights between 0 and 1
        )

        # Two branches with the same block
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            self.block
        )

        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            self.block
        )

        # Concatenate the outputs of both branches
        self.concat = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features='mask', out_features=10)
        )

    def forward(self, x):
        # the initial shape of x is: (1, 3, 32, 32)
        # Split the input into two branches
        x1 = self.branch1(x)  # output tensor shape: (1, 512)
        x2 = self.branch2(x)  # output tensor shape: (1, 512)

        # Element-wise multiplication with the input (assuming x is the input to the block)
        # Here, we need to adjust the dimensions properly
        # The block output is of shape (batch_size, 512, 1, 1)
        # We need to reshape it to (batch_size, 512, 32, 32) to match the input dimensions
        # But since the input is 3x32x32, and the block output is 512x1x1, we need to reshape the block output to 512x32x32
        # However, the block output is a weight, so we need to reshape it to match the input's spatial dimensions
        # For simplicity, we'll assume the input is 3x32x32, and the block output is 512x1x1, which is reshaped to 512x32x32
        # But this is not correct because the block output is a weight for the input's channels (3), not the spatial dimensions
        # So, the correct approach is to reshape the block output to (batch_size, 3, 1, 1) to match the input's channel dimension
        # Then, perform element-wise multiplication with the input's channel dimension
        # However, the block output is 512 channels, not 3, so this approach is incorrect
        # The correct approach is to reshape the block output to (batch_size, 512, 1, 1) and then multiply with the input's channel dimension
        # But the input has 3 channels, so this doesn't align
        # This suggests that the block's input should be the output of the Conv2d layer, which has 512 channels
        # So, the block's input is 512x32x32, and the block's output is 512x1x1, which is reshaped to 512x32x32
        # Then, element-wise multiplication is performed between the block's output and the input to the block (512x32x32)
        # So, the block's output is a weight for the input's spatial dimensions
        # This seems to be the intended design

        # For branch1
        conv_out1 = self.branch1[0](x)  # output tensor shape: (1, 512, 32, 32)
        block_out1 = self.block(conv_out1)  # output tensor shape: (1, 512)
        block_out1 = block_out1.view(-1, 512, 1, 1)  # output tensor shape: (1, 512, 1, 1)
        weighted_out1 = conv_out1 * block_out1  # output tensor shape: (1, 512, 32, 32)

        # For branch2
        conv_out2 = self.branch2[0](x)  # output tensor shape: (1, 512, 32, 32)
        block_out2 = self.block(conv_out2)  # output tensor shape: (1, 512)
        block_out2 = block_out2.view(-1, 512, 1, 1)  # output tensor shape: (1, 512, 1, 1)
        weighted_out2 = conv_out2 * block_out2  # output tensor shape: (1, 512, 32, 32)

        # Concatenate the outputs of both branches
        concat_out = torch.cat((weighted_out1, weighted_out2), dim=1)  # output tensor shape: (1, 1024, 32, 32)
        concat_out = self.concat(concat_out)

        return concat_out

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 3, 32, 32)  # Dummy input tensor for CIFAR-10 (batch_size=1, channels=3, height=32, width=32)
    output_tensor = model(input_tensor)
    print("Output tensor shape:", output_tensor.shape)