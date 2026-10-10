import torch
import torch.nn as nn
import torch.nn.functional as F

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()

        # First block: three max pooling layers with different scales
        self.max_pool1 = nn.MaxPool2d(kernel_size=1, stride=1, padding=0)
        self.max_pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.max_pool3 = nn.MaxPool2d(kernel_size=4, stride=4, padding=0)

        # Flatten layers for each pooling output
        self.flatten1 = nn.Flatten()
        self.flatten2 = nn.Flatten()
        self.flatten3 = nn.Flatten()

        # Dropout layer
        self.dropout = nn.Dropout(p=0.5)

        # Fully connected layer and reshape for second block
        self.fc1 = nn.Linear(3*32*32*3, 1024)  # Assuming input is 32x32x3 (CIFAR-10)
        self.unflatten = nn.Unflatten(1, (32, 32, 32))  # Reshape to 32x32x32

        # Second block: separable convolutions with different kernel sizes
        self.sep_conv1 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=1, stride=1, padding=0),
            nn.ReLU()
        )
        self.sep_conv2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        self.sep_conv3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, stride=1, padding=2),
            nn.ReLU()
        )
        self.sep_conv4 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=7, stride=1, padding=3),
            nn.ReLU()
        )

        # Final classification layers
        self.fc2 = nn.Linear(32*32*32, 10)  # 10 classes for CIFAR-10

    def forward(self, x):
        # First block processing
        pool1 = self.max_pool1(x)
        pool2 = self.max_pool2(x)
        pool3 = self.max_pool3(x)

        # Flatten each pooling output
        flat1 = self.flatten1(pool1)
        flat2 = self.flatten2(pool2)
        flat3 = self.flatten3(pool3)

        # Concatenate with dropout
        concat = torch.cat([flat1, flat2, flat3], dim=1)
        concat = self.dropout(concat)

        # Process through fully connected and reshape
        fc_out = self.fc1(concat)
        reshaped = self.unflatten(fc_out)

        # Second block processing
        # Split into four groups along last dimension
        split = torch.split(reshaped, reshaped.size(3)//4, dim=3)

        # Process each group with separable convolutions
        conv1 = self.sep_conv1(split[0])
        conv2 = self.sep_conv2(split[1])
        conv3 = self.sep_conv3(split[2])
        conv4 = self.sep_conv4(split[3])

        # Concatenate outputs
        concat_conv = torch.cat([conv1, conv2, conv3, conv4], dim=1)

        # Final classification
        flat_final = self.flatten1(concat_conv)
        output = self.fc2(flat_final)

        return output

if __name__ == "__main__":
    model = DL_Model()
    # Create a dummy input tensor with shape (batch_size, channels, height, width)
    # CIFAR-10 images are 32x32 with 3 channels
    input_tensor = torch.randn(1, 3, 32, 32)  # batch_size=1
    output_tensor = model(input_tensor)
    print("Input shape:", input_tensor.shape)
    print("Output shape:", output_tensor.shape)