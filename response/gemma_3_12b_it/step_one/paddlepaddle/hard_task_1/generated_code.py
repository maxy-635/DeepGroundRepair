import paddle
import paddle.nn as nn
import paddle.nn.functional as F

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Initial convolutional layer
        self.conv1 = nn.Conv2D(in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=1)

        # Block 1: Channel Attention
        self.global_avg_pool = nn.AdaptiveAvgPool2D(output_size=1)
        self.global_max_pool = nn.AdaptiveMaxPool2D(output_size=1)
        self.fc1_avg = nn.Linear(in_features=3, out_features=3)
        self.fc2_avg = nn.Linear(in_features=3, out_features=3)
        self.fc1_max = nn.Linear(in_features=3, out_features=3)
        self.fc2_max = nn.Linear(in_features=3, out_features=3)

        # Block 2: Spatial Attention
        self.avg_pool = nn.AdaptiveAvgPool2D(output_size=1)
        self.max_pool = nn.AdaptiveMaxPool2D(output_size=1)
        self.conv_spatial = nn.Conv2D(in_channels=2, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.sigmoid = nn.Sigmoid()

        # Additional branch
        self.conv_additional = nn.Conv2D(in_channels=3, out_channels=3, kernel_size=1, stride=1, padding=0)

        # Final classification
        self.fc_classifier = nn.Linear(in_features=3, out_features=10)

    def forward(self, x):
        # Initial convolutional layer
        x = self.conv1(x)

        # Block 1: Channel Attention
        avg_out = self.global_avg_pool(x)
        avg_out = self.fc1_avg(avg_out.reshape([avg_out.shape[0], -1]))
        avg_out = F.relu(avg_out)
        avg_out = self.fc2_avg(avg_out)

        max_out = self.global_max_pool(x)
        max_out = self.fc1_max(max_out.reshape([max_out.shape[0], -1]))
        max_out = F.relu(max_out)
        max_out = self.fc2_max(max_out)

        channel_attention = avg_out + max_out
        channel_attention = F.sigmoid(channel_attention)
        channel_attention = channel_attention.reshape([channel_attention.shape[0], 3, 1, 1])
        x = x * channel_attention

        # Block 2: Spatial Attention
        avg_spatial = self.avg_pool(x)
        max_spatial = self.max_pool(x)
        spatial_concat = paddle.concat([avg_spatial, max_spatial], axis=1)
        spatial_attention = self.conv_spatial(spatial_concat)
        spatial_attention = self.sigmoid(spatial_attention)
        x = x * spatial_attention

        # Additional branch
        additional_branch = self.conv_additional(x)
        x = x + additional_branch

        # Final classification
        x = F.flatten(x, start=1)
        x = self.fc_classifier(x)

        return x


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.rand(shape=[1, 3, 32, 32])
    output_tensor = model(input_tensor)
    print(output_tensor.shape)