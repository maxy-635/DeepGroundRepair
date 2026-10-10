import torch
import torch.nn as nn

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()
        # First block: 3 average pooling layers with different scales
        self.pool1 = nn.AvgPool2d(kernel_size=(1, 1), stride=(1, 1))
        self.pool2 = nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.pool3 = nn.AvgPool2d(kernel_size=(4, 4), stride=(4, 4))

        # Flatten the pooled outputs
        self.flatten = nn.Flatten()

        # Fully connected layer and reshape for second block
        self.fc = nn.Linear(28 * 28 * 3, 4)  # Assuming input is 28x28x3 after pooling

        # Second block: Depthwise separable convolutions
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 1, kernel_size=(1, 1), stride=(1, 1)),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(3, 1, kernel_size=(3, 3), stride=(1, 1)),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(3, 1, kernel_size=(5, 5), stride=(1, 1)),
            nn.ReLU(inplace=True)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(3, 1, kernel_size=(7, 7), stride=(1, 1)),
            nn.ReLU(inplace=True)
        )

        # Concatenate the outputs of the depthwise convolutions
        self.concat = nn.Sequential(
            nn.Unflatten(1, (1, 1, 1)),  # Adjust dimensions as needed
            nn.Linear(4, 10)  # Assuming 10 classes for MNIST
        )

    def forward(self, x):
        # First block
        x1 = self.pool1(x)
        x2 = self.pool2(x)
        x3 = self.pool3(x)

        # Flatten and concatenate the pooled outputs
        x1_flat = self.flatten(x1)
        x2_flat = self.flatten(x2)
        x3_flat = self.flatten(x3)

        # Concatenate the flattened outputs
        x_concat = torch.cat((x1_flat, x2_flat, x3_flat), dim=1)

        # Reshape and pass through fully connected layer
        x_reshape = x_concat.view(-1, 4)  # Assuming reshaping to 4D tensor
        x_fc = self.fc(x_reshape)

        # Second block
        x_conv1 = self.conv1(x)
        x_conv2 = self.conv2(x)
        x_conv3 = self.conv3(x)
        x_conv4 = self.conv4(x)

        # Concatenate the outputs of the depthwise convolutions
        x_conv_concat = torch.cat((x_conv1, x_conv2, x_conv3, x_conv4), dim=1)

        # Flatten and pass through final fully connected layer
        x_conv_flat = self.flatten(x_conv_concat)
        x_final = self.concat(x_conv_flat)

        return x_final

if __name__ == "__main__":
    # Create a dummy input tensor with shape (batch_size, channels, height, width)
    input_tensor = torch.randn(1, 3, 28, 28)
    model = DL_Model()
    output_tensor = model(input_tensor)
    print("Input shape:", input_tensor.shape)
    print("Output shape:", output_tensor.shape)