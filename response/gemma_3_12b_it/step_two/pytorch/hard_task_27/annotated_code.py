import torch
import torch.nn as nn

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()
        self.conv = nn.Conv2d(
            in_channels=3,
            out_channels=3,
            kernel_size=7,
            stride=1,
            padding=3
        )
        self.layer_norm = nn.LayerNorm(normalized_shape='mask')
        self.fc1 = nn.Linear(
            in_features=3 * 32 * 32,
            out_features=3 * 32 * 32
        )
        self.fc2 = nn.Linear(
            in_features=3 * 32 * 32,
            out_features=10
        )

    def forward(self, x):
        # the initial shape of x is: (1, 3, 32, 32)
        x = self.conv(x)  # output tensor shape: (1, 3, 32, 32)
        x = self.layer_norm(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        return x

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 3, 32, 32)
    output_tensor = model(input_tensor)
    print(output_tensor.shape)