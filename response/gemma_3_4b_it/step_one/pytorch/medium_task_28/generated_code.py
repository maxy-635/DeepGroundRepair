import torch
import torch.nn as nn

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()
        self.attention = nn.Sequential(
            nn.Conv1d(3, 32, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv1d(32, 32, kernel_size=1, stride=1, padding=0),
            nn.Softmax(dim=1)
        )
        self.reduce = nn.Sequential(
            nn.Conv1d(32, 8, kernel_size=1, stride=1, padding=0),
            nn.LayerNorm(8),
            nn.ReLU(inplace=True),
            nn.Conv1d(8, 32, kernel_size=1, stride=1, padding=0)
        )
        self.linear = nn.Linear(32, 10)

    def forward(self, x):
        context = self.attention(x)
        x = x * context
        x = self.reduce(x)
        x = x + x
        x = x.flatten(1)
        x = self.linear(x)
        return x

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 3, 32)  # Batch size 1, 3 channels, 32 spatial dimensions
    output_tensor = model(input_tensor)
    print(output_tensor.shape)