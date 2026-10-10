import paddle
import paddle.nn as nn

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()
        # Initial 1x1 convolution to increase channels threefold
        self.conv1 = nn.Conv2D(in_channels=3,
                                 out_channels=9,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)

        # 3x3 Depthwise Separable Convolution
        self.depthwise = nn.Conv2D(in_channels=9,
                                     out_channels=9,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1,
                                     groups=9)
        self.pointwise = nn.Conv2D(in_channels=9,
                                     out_channels=9,
                                     kernel_size=1,
                                     stride=1,
                                     padding=0)

        # Channel Attention Module
        self.avg_pool = nn.AdaptiveAvgPool2D(output_size=1)
        self.fc1 = nn.Linear(in_features=9, out_features=9)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(in_features=9, out_features=9)

        # 1x1 Convolution to reduce dimensionality
        self.conv_reduce = nn.Conv2D(in_channels=18,
                                       out_channels=3,
                                       kernel_size=1,
                                       stride=1,
                                       padding=0)

        # Flatten and Fully Connected Layer for Classification
        self.flatten = nn.Flatten()
        self.fc_classifier = nn.Linear(in_features=3 * 32 * 32, out_features=10)

    def forward(self, x):
        # Initial 1x1 convolution
        x = self.conv1(x)

        # Depthwise Separable Convolution
        x = self.depthwise(x)
        x = self.pointwise(x)

        # Channel Attention
        attn = self.avg_pool(x)
        attn = self.fc1(attn)
        attn = self.relu(attn)
        attn = self.fc2(attn)
        attn = paddle.reshape(attn, [x.shape[0], -1])
        attn = paddle.reshape(attn, [x.shape[0], 9, 1, 1])
        x = x * attn

        # Combine with initial input
        x = paddle.concat([x, self.conv1(x)], axis=1)

        # 1x1 Convolution to reduce dimensionality
        x = self.conv_reduce(x)

        # Flatten and Classification
        x = self.flatten(x)
        x = self.fc_classifier(x)

        return x


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn(shape=[1, 3, 32, 32])
    output_tensor = model(input_tensor)
    print(output_tensor.shape)