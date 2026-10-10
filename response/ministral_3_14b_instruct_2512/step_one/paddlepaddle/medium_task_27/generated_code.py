import paddle
import paddle.nn as nn
import paddle.nn.functional as F

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Branch 1: 3x3 convolution
        self.branch1_conv1 = nn.Conv2D(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.branch1_conv2 = nn.Conv2D(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.branch1_pool = nn.MaxPool2D(kernel_size=2, stride=2, padding=0)

        # Branch 2: 5x5 convolution
        self.branch2_conv1 = nn.Conv2D(in_channels=3, out_channels=64, kernel_size=5, stride=1, padding=2)
        self.branch2_conv2 = nn.Conv2D(in_channels=64, out_channels=64, kernel_size=5, stride=1, padding=2)
        self.branch2_pool = nn.MaxPool2D(kernel_size=2, stride=2, padding=0)

        # Attention mechanism
        self.attention_conv = nn.Conv2D(in_channels=128, out_channels=128, kernel_size=1, stride=1, padding=0)
        self.attention_pool = nn.AdaptiveAvgPool2D(output_size=(1, 1))
        self.attention_fc1 = nn.Linear(in_features=128, out_features=64)
        self.attention_fc2 = nn.Linear(in_features=64, out_features=2)
        self.attention_softmax = nn.Softmax(axis=1)

        # Final layers
        self.final_conv = nn.Conv2D(in_channels=128, out_channels=128, kernel_size=1, stride=1, padding=0)
        self.final_pool = nn.AdaptiveAvgPool2D(output_size=(1, 1))
        self.final_fc = nn.Linear(in_features=128, out_features=10)

        # Activation functions
        self.relu = nn.ReLU()

    def forward(self, x):
        # Branch 1 processing
        branch1 = self.branch1_conv1(x)
        branch1 = self.relu(branch1)
        branch1 = self.branch1_conv2(branch1)
        branch1 = self.relu(branch1)
        branch1 = self.branch1_pool(branch1)

        # Branch 2 processing
        branch2 = self.branch2_conv1(x)
        branch2 = self.relu(branch2)
        branch2 = self.branch2_conv2(branch2)
        branch2 = self.relu(branch2)
        branch2 = self.branch2_pool(branch2)

        # Combine branches
        combined = paddle.concat([branch1, branch2], axis=1)

        # Attention mechanism
        attention_features = self.attention_conv(combined)
        attention_features = self.attention_pool(attention_features)
        attention_features = paddle.reshape(attention_features, [attention_features.shape[0], -1])
        attention_weights = self.attention_fc1(attention_features)
        attention_weights = self.relu(attention_weights)
        attention_weights = self.attention_fc2(attention_weights)
        attention_weights = self.attention_softmax(attention_weights)

        # Weighted combination
        branch1_weighted = paddle.multiply(branch1, attention_weights[:, 0:1])
        branch2_weighted = paddle.multiply(branch2, attention_weights[:, 1:2])
        weighted_output = branch1_weighted + branch2_weighted

        # Final processing
        final_features = self.final_conv(weighted_output)
        final_features = self.final_pool(final_features)
        final_features = paddle.reshape(final_features, [final_features.shape[0], -1])
        output = self.final_fc(final_features)

        return output

if __name__ == "__main__":
    model = DL_Model()
    # Create a dummy input tensor with shape [batch_size, 3, 32, 32] for CIFAR-10
    input_tensor = paddle.randn((1, 3, 32, 32))
    output_tensor = model(input_tensor)
    print("Input shape:", input_tensor.shape)
    print("Output shape:", output_tensor.shape)