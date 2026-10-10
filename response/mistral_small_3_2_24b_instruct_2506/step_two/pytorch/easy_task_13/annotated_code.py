import torch
import torch.nn as nn

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()
        # First 1x1 convolutional layer
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=1, stride=1, padding=0)
        self.dropout1 = nn.Dropout(p=0.5)

        # Second 1x1 convolutional layer
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, stride=1, padding=0)
        self.dropout2 = nn.Dropout(p=0.5)

        # 3x1 convolutional layer
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 1), stride=1, padding=0)
        self.dropout3 = nn.Dropout(p=0.5)

        # 1x3 convolutional layer
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(1, 3), stride=1, padding=0)
        self.dropout4 = nn.Dropout(p=0.5)

        # Restore channels to input's channel count (1) using another 1x1 convolutional layer
        self.conv5 = nn.Conv2d(in_channels=64, out_channels=1, kernel_size='mask', stride='mask', padding='mask')
        self.dropout5 = nn.Dropout(p=0.5)

        # Flattening layer
        self.flatten = nn.Flatten()

        # Fully connected layer
        self.fc = nn.Linear(in_features=28*28, out_features=10)

    def forward(self, x):
        # the initial shape of x is: (1, 1, 28, 28)
        # Save the original input
        original_input = x  # output tensor shape: (1, 1, 28, 28)

        # Apply the first 1x1 convolutional layer with dropout
        x = self.conv1(x)  # output tensor shape: (1, 64, 28, 28)
        x = self.dropout1(x)  # output tensor shape: (1, 64, 28, 28)

        # Apply the second 1x1 convolutional layer with dropout
        x = self.conv2(x)  # output tensor shape: (1, 64, 28, 28)
        x = self.dropout2(x)  # output tensor shape: (1, 64, 28, 28)

        # Apply the 3x1 convolutional layer with dropout
        x = self.conv3(x)  # output tensor shape: (1, 64, 26, 28)
        x = self.dropout3(x)  # output tensor shape: (1, 64, 26, 28)

        # Apply the 1x3 convolutional layer with dropout
        x = self.conv4(x)  # output tensor shape: (1, 64, 26, 26)
        x = self.dropout4(x)  # output tensor shape: (1, 64, 26, 26)

        # Apply the final 1x1 convolutional layer to restore channels
        x = self.conv5(x)  # output tensor shape of this layer should be (1, 1, 28, 28)
        x = self.dropout5(x)

        # Combine with the original input via addition
        x = x + original_input

        # Flatten the output
        x = self.flatten(x)

        # Apply the fully connected layer
        x = self.fc(x)

        return x

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 1, 28, 28)  # Dummy input tensor for MNIST (batch_size=1, channels=1, height=28, width=28)
    output_tensor = model(input_tensor)
    print(output_tensor)