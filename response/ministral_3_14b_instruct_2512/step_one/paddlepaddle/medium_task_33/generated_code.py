import paddle
import paddle.nn as nn
import paddle.nn.functional as F

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Channel group 1: 1x1 separable convolution
        self.conv1x1_group1 = nn.Sequential(
            nn.Conv2D(in_channels=3, out_channels=64, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2D(num_features=64),
            nn.ReLU(),
            nn.MaxPool2D(kernel_size=2, stride=2, padding=0)
        )

        # Channel group 2: 3x3 separable convolution
        self.conv3x3_group2 = nn.Sequential(
            nn.Conv2D(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2D(num_features=64),
            nn.ReLU(),
            nn.MaxPool2D(kernel_size=2, stride=2, padding=0)
        )

        # Channel group 3: 5x5 separable convolution
        self.conv5x5_group3 = nn.Sequential(
            nn.Conv2D(in_channels=3, out_channels=64, out_channels=64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2D(num_features=64),
            nn.ReLU(),
            nn.MaxPool2D(kernel_size=2, stride=2, padding=0)
        )

        # Concatenate the outputs from all three groups
        self.concat = nn.Sequential()

        # Fully connected layers
        self.fc1 = nn.Linear(in_features=64*8*8, out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=256)
        self.fc3 = nn.Linear(in_features=256, out_features=10)  # CIFAR-10 has 10 classes

        # Dropout for regularization
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        # Process each channel group
        x1 = self.conv1x1_group1(x)
        x2 = self.conv3x3_group2(x)
        x3 = self.conv5x5_group3(x)

        # Concatenate along channel dimension
        x = paddle.concat([x1, x2, x3], axis=1)

        # Flatten the features
        x = nn.Flatten()(x)

        # Fully connected layers with dropout
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.fc3(x)

        return x

if __name__ == "__main__":
    model = DL_Model()
    # Create a dummy input tensor with shape [batch_size, 3, 32, 32] for CIFAR-10
    input_tensor = paddle.randn((1, 3, 32, 32))
    output_tensor = model(input_tensor)
    print("Input shape:", input_tensor.shape)
    print("Output shape:", output_tensor.shape)