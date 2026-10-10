import paddle
import paddle.nn as nn

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()
        # Split the input into three groups along the channel dimension
        self.conv_group1 = nn.Sequential(
            nn.Conv2D(in_channels=3, out_channels=32, kernel_size=1, stride=1, padding=0),
            nn.Conv2D(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.Conv2D(in_channels=32, out_channels=32, kernel_size=1, stride=1, padding=0)
        )

        self.conv_group2 = nn.Sequential(
            nn.Conv2D(in_channels=3, out_channels=32, kernel_size=1, stride=1, padding=0),
            nn.Conv2D(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.Conv2D(in_channels=32, out_channels=32, kernel_size=1, stride=1, padding=0)
        )

        self.conv_group3 = nn.Sequential(
            nn.Conv2D(in_channels=3, out_channels=32, kernel_size=1, stride=1, padding=0),
            nn.Conv2D(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.Conv2D(in_channels=32, out_channels=32, kernel_size=1, stride=1, padding=0)
        )

        # Main path
        self.main_path = nn.Sequential(
            nn.Conv2D(in_channels=3, out_channels=32, kernel_size=1, stride=1, padding=0),
            nn.Conv2D(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.Conv2D(in_channels=32, out_channels=32, kernel_size=1, stride=1, padding=0)
        )

        # Layer normalization
        self.layer_norm = nn.LayerNorm(normalized_shape=32)

        # Flatten and fully connected layer
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(in_features=32 * 32 * 32, out_features=10)

    def forward(self, x):
        # Split the input into three groups along the channel dimension
        group1 = self.conv_group1(x)
        group2 = self.conv_group2(x)
        group3 = self.conv_group3(x)

        # Combine the outputs from the three groups
        combined = paddle.add(group1, paddle.add(group2, group3))

        # Main path
        main_path = self.main_path(x)
        # Add the main path with the combined features
        fused_features = paddle.add(combined, main_path)

        # Apply layer normalization
        fused_features = self.layer_norm(fused_features)

        # Flatten and pass through fully connected layer
        flattened = self.flatten(fused_features)
        output = self.fc(flattened)

        return output

if __name__ == "__main__":
    # Create a dummy input tensor with shape (batch_size, channels, height, width)
    input_tensor = paddle.randn((1, 3, 32, 32))
    model = DL_Model()
    output_tensor = model(input_tensor)
    print("Input shape:", input_tensor.shape)
    print("Output shape:", output_tensor.shape)