import paddle
import paddle.nn as nn
import paddle.nn.functional as F

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Branch 1: 1x1 conv + 3x3 conv
        self.branch1 = nn.Sequential(
            nn.Conv2D(in_channels=3, out_channels=64, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2D(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Dropout(p=0.2)
        )

        # Branch 2: 1x1 conv + 1x7 conv + 7x1 conv + 3x3 conv
        self.branch2 = nn.Sequential(
            nn.Conv2D(in_channels=3, out_channels=64, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2D(in_channels=64, out_channels=64, kernel_size=(1, 7), stride=1, padding=(0, 3)),
            nn.ReLU(),
            nn.Conv2D(in_channels=64, out_channels=64, kernel_size=(7, 1), stride=1, padding=(3, 0)),
            nn.ReLU(),
            nn.Conv2D(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Dropout(p=0.2)
        )

        # Branch 3: Max pooling
        self.branch3 = nn.Sequential(
            nn.MaxPool2D(kernel_size=3, stride=1, padding=1),
            nn.Dropout(p=0.2)
        )

        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(in_features='mask', out_features=512),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=512, out_features=512),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=512, out_features=10)
        )

    def forward(self, x):
        # the initial shape of x is: (1, 3, 32, 32)
        # Process each branch
        branch1_out = self.branch1(x)  # output tensor shape: (1, 64, 32, 32)
        branch2_out = self.branch2(x)  # output tensor shape: (1, 64, 32, 32)
        branch3_out = self.branch3(x)  # output tensor shape: (1, 3, 32, 32)

        # Concatenate outputs from all branches
        concat_out = paddle.concat([branch1_out, branch2_out, branch3_out], axis=1)  # output tensor shape: (1, 131, 32, 32)

        # Flatten for fully connected layers
        flatten_out = paddle.reshape(concat_out, [concat_out.shape[0], -1])  # output tensor shape: (1, 134144)

        # Process through fully connected layers
        out = self.fc(flatten_out)

        return out

if __name__ == "__main__":
    model = DL_Model()
    # Create a dummy input tensor with shape [batch_size, 3, 32, 32] (CIFAR-10 image size)
    input_tensor = paddle.randn((1, 3, 32, 32))
    output_tensor = model(input_tensor)
    print("Input shape:", input_tensor.shape)
    print("Output shape:", output_tensor.shape)