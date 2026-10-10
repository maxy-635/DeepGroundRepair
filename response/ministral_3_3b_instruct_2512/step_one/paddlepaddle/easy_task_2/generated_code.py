import paddle
import paddle.nn as nn

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()
        # Sequential feature extraction layers (2 layers)
        self.feature_extraction_layers = nn.Sequential(
            nn.Conv2D(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.AvgPool2D(kernel_size=2, stride=2, padding=0),
            nn.Conv2D(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.AvgPool2D(kernel_size=2, stride=2, padding=0)
        )

        # Additional convolutional layers and pooling (3 layers + 1 pooling)
        self.additional_layers = nn.Sequential(
            nn.Conv2D(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.Conv2D(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.Conv2D(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=1),
            nn.AvgPool2D(kernel_size=2, stride=2, padding=0)
        )

        # Flatten and fully connected layers with dropout
        self.flatten = nn.Flatten()
        self.fc_layers = nn.Sequential(
            nn.Linear(in_features=1024 * 7 * 7, out_features=512),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=512, out_features=256),
            nn.Dropout(p=0.5)
        )

        # Softmax layer for classification
        self.softmax = nn.Linear(in_features=256, out_features=1000)

    def forward(self, x):
        x = self.feature_extraction_layers(x)
        x = self.additional_layers(x)
        x = self.flatten(x)
        x = self.fc_layers(x)
        logits = self.softmax(x)
        return paddle.nn.functional.softmax_with_cross_entropy(logits, label=x)

if __name__ == "__main__":
    model = DL_Model()
    # Create a dummy input tensor with shape (batch_size, channels, height, width)
    input_tensor = paddle.randn((1, 3, 224, 224))
    output_tensor = model(input_tensor)
    print(output_tensor.shape)