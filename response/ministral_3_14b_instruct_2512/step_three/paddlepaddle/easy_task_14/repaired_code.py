import paddle
import paddle.nn as nn
import paddle.nn.functional as F

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()
        # Define the convolutional layers
        self.conv1 = nn.Conv2D(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2D(num_features=64)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2D(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2D(num_features=128)
        self.relu2 = nn.ReLU()

        self.conv3 = nn.Conv2D(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2D(num_features=256)
        self.relu3 = nn.ReLU()

        # Global average pooling
        self.avg_pool = nn.AdaptiveAvgPool2D(output_size=(1, 1))

        # Fully connected layers for generating weights
        self.fc1 = nn.Linear(in_features=256, out_features=256)
        self.fc2 = nn.Linear(in_features=256, out_features=256)

        # Final fully connected layer for classification
        self.fc3 = nn.Linear(in_features=256, out_features=10)

        # Layer normalization for the weights
        self.ln = nn.LayerNorm(normalized_shape=[256])

    def forward(self, x):
        # Initial convolutional layers
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)

        # Global average pooling
        x = self.avg_pool(x)

        # Flatten the pooled features
        x = paddle.flatten(x, start_axis=1)

        # Generate weights using fully connected layers
        weights = self.fc1(x)
        weights = self.fc2(weights)
        weights = self.ln(weights)
        weights = paddle.unsqueeze(weights, axis=1)
        weights = paddle.unsqueeze(weights, axis=2)

        # Reshape the input feature map to align with weights
        input_features = paddle.flatten(x, start_axis=1)
        input_features = paddle.unsqueeze(input_features, axis=1)
        input_features = paddle.unsqueeze(input_features, axis=2)

        # Element-wise multiplication
        x = input_features * weights

        # Flatten and pass through the final fully connected layer
        x = paddle.flatten(x, start_axis=1)
        x = self.fc3(x)

        return x

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn((1, 3, 32, 32))  # Batch size 1, 3 channels, 32x32 image
    output_tensor = model(input_tensor)
    print(output_tensor.shape)