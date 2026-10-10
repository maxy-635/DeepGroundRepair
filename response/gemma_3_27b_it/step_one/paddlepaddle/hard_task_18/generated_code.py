import paddle
import paddle.nn as nn
import paddle.distribution as dist

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()

        # First block: Feature extraction
        self.block1 = nn.Sequential(
            nn.Conv2D(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2D(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AvgPool2D(kernel_size=2, stride=2, padding=0)
        )

        # Second block: Channel weight generation and application
        self.global_pool = nn.AdaptiveAvgPool2D((1, 1))
        self.fc1 = nn.Linear(in_features=32, out_features=32)
        self.fc2 = nn.Linear(in_features=32, out_features=32)
        self.reshape = dist.ReshapeTransform(in_event_shape=[32], out_event_shape=[32, 1, 1])
        
        # Classification layer
        self.flatten = nn.Flatten()
        self.fc_classifier = nn.Linear(in_features=32 * 32 * 32, out_features=10)
        

    def forward(self, x):
        # First block
        x1 = self.block1(x)

        # Second block
        x_gap = self.global_pool(x1)
        x_gap = self.flatten(x_gap)
        x_gap = self.fc1(x_gap)
        x_gap = self.fc2(x_gap)
        x_gap = self.reshape(x_gap)

        # Apply channel weights
        x = x * x_gap

        # Classification
        x = self.flatten(x)
        x = self.fc_classifier(x)
        return x


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn([1, 3, 32, 32])  # Example input tensor
    output_tensor = model(input_tensor)
    print(output_tensor.shape)