import torch
import torch.nn as nn

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Define the block
        def block(input_tensor):
            # Global average pooling
            pooled_output = nn.AdaptiveAvgPool2d(output_size=1)(input_tensor)

            # Fully connected layers to generate weights
            weight_layer_1 = nn.Linear(in_features=pooled_output.shape[1], out_features=pooled_output.shape[1])
            weight_layer_2 = nn.Linear(in_features=pooled_output.shape[1], out_features=pooled_output.shape[1])

            weights_1 = weight_layer_1(pooled_output)
            weights_2 = weight_layer_2(weights_1)
            
            # Reshape weights to match input shape
            weights = weights_2.reshape(input_tensor.shape[0], -1, 1, 1)

            # Element-wise multiplication
            output = input_tensor * weights
            return output

        # Two branches with the same block
        self.branch_1 = block
        self.branch_2 = block

        # Concatenate outputs from both branches
        self.concat = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=1),
            nn.Flatten(),
            nn.Linear(in_features=64 * 32 * 32, out_features=10)
        )

    def forward(self, x):
        # Pass the input through both branches
        output_branch_1 = self.branch_1(x)
        output_branch_2 = self.branch_2(x)

        # Concatenate the outputs
        concatenated_output = torch.cat((output_branch_1, output_branch_2), dim=1)

        # Flatten and pass through the final fully connected layer
        output = self.concat(concatenated_output)
        return output


if __name__ == "__main__":
    model = DL_Model()
    # Dummy input tensor for CIFAR-10 (32x32 RGB image)
    input_tensor = torch.randn(1, 3, 32, 32)
    output_tensor = model(input_tensor)
    print(output_tensor.shape)