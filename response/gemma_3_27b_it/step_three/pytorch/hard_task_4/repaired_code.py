import torch
import torch.nn as nn
import torch.nn.functional as F

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Channel expansion
        self.channel_expand = nn.Conv2d(in_channels=3, out_channels=9, kernel_size=1, stride=1, padding=0)

        # Depthwise separable convolution
        self.depthwise_conv = nn.Conv2d(in_channels=9, out_channels=9, kernel_size=3, stride=1, padding=1, groups=9)

        # Channel attention
        self.global_avg_pool = nn.AvgPool2d(kernel_size=7, stride=7)
        self.fc1 = nn.Linear(in_features=144, out_features=9 // 4)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(in_features=9 // 4, out_features=9)
        self.sigmoid = nn.Sigmoid()

        # Dimensionality reduction
        self.channel_reduce = nn.Conv2d(in_channels=9, out_channels=3, kernel_size=1, stride=1, padding=0)

        # Residual connection
        self.residual_connection = lambda x, y: x + y

        # Classification layers
        self.flatten = nn.Flatten()
        self.fc_classifier = nn.Linear(in_features=3 * 32 * 32, out_features=10)

    def forward(self, x):
        # the initial shape of x is: (1, 3, 32, 32)
        # Channel expansion
        x = self.channel_expand(x)  # output tensor shape: (1, 9, 32, 32)

        # Depthwise separable convolution
        x = self.depthwise_conv(x)  # output tensor shape: (1, 9, 32, 32)

        # Channel attention
        attention = self.global_avg_pool(x)  # output tensor shape: (1, 9, 4, 4)
        attention = self.flatten(attention)  # output tensor shape: (1, 144)
        attention = self.fc1(attention)
        attention = self.relu(attention)
        attention = self.fc2(attention)
        attention = self.sigmoid(attention)
        attention = attention.reshape(x.shape[1], 1, 1)
        x = x * attention

        # Dimensionality reduction
        x = self.channel_reduce(x)

        # Residual connection
        x = self.residual_connection(x, self.channel_expand(x))

        # Classification
        x = self.flatten(x)
        x = self.fc_classifier(x)

        return x


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 3, 32, 32)
    output_tensor = model(input_tensor)
    print(output_tensor.shape)