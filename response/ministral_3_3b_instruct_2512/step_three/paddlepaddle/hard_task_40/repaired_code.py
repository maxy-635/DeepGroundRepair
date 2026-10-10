import paddle
import paddle.nn as nn

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()
        # First Block: Average Pooling and Flattening
        self.avg_pool1 = nn.AvgPool2D(kernel_size=1, stride=1)
        self.avg_pool2 = nn.AvgPool2D(kernel_size=2, stride=2)
        self.avg_pool3 = nn.AvgPool2D(kernel_size=4, stride=4)

        # Flatten layers
        self.flatten = nn.Flatten()

        # Fully connected layer and reshape
        self.fc = nn.Linear(144, 4)  # Assuming input is 28x28, flattened to 784, then pooled to 144
        self.reshape = nn.Flatten()

        # Second Block: Parallel Paths for Multi-scale Feature Extraction
        self.path1 = nn.Sequential(
            nn.Conv2D(in_channels=1, out_channels=32, kernel_size=1, stride=1, padding=0),
            nn.Dropout(p=0.5)
        )

        self.path2 = nn.Sequential(
            nn.Conv2D(in_channels=1, out_channels=32, kernel_size=1, stride=1, padding=0),
            nn.Conv2D(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.Conv2D(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.Dropout(p=0.5)
        )

        self.path3 = nn.Sequential(
            nn.Conv2D(in_channels=1, out_channels=32, kernel_size=1, stride=1, padding=0),
            nn.Conv2D(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.Dropout(p=0.5)
        )

        self.path4 = nn.Sequential(
            nn.AvgPool2D(kernel_size=1, stride=1),
            nn.Conv2D(in_channels=1, out_channels=32, kernel_size=1, stride=1, padding=0),
            nn.Dropout(p=0.5)
        )

        # Concatenate paths
        self.concat = nn.Sequential(
            nn.Reshape((-1, 32, 4)),  # Adjust shape to concatenate along channel dimension
            nn.Linear(128, 10)  # Assuming concatenated features are reshaped to 128
        )

        # Final fully connected layers
        self.fc1 = nn.Linear(128, 10)

    def forward(self, x):
        # First Block: Average Pooling and Flattening
        x1 = self.avg_pool1(x)
        x2 = self.avg_pool2(x)
        x3 = self.avg_pool3(x)

        # Flatten and concatenate outputs
        x1_flat = self.flatten(x1)
        x2_flat = self.flatten(x2)
        x3_flat = self.flatten(x3)

        concatenated = paddle.concat([x1_flat, x2_flat, x3_flat], axis=1)

        # Reshape and pass through first FC layer
        x = self.reshape(concatenated)
        x = self.fc(x)

        # Second Block: Parallel Paths
        path1_out = self.path1(x)
        path2_out = self.path2(x)
        path3_out = self.path3(x)
        path4_out = self.path4(x)

        # Concatenate outputs from all paths
        concatenated_paths = paddle.concat([path1_out, path2_out, path3_out, path4_out], axis=1)

        # Final classification
        output = self.fc1(concatenated_paths)
        return output

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.to_tensor([[[[1, 2, 3], [4, 5, 6], [7, 8, 9]]]])  # Dummy input tensor for MNIST (28x28)
    output_tensor = model(input_tensor)
    print(output_tensor)