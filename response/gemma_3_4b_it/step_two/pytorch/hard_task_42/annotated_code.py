import torch
import torch.nn as nn

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Block 1
        self.block1 = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=0),
            nn.Conv1d(32, 64, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=4, stride=4, padding=0),
            nn.Conv1d(64, 128, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=1, stride=1, padding=0)
        )

        # Block 2
        self.block2 = nn.Sequential(
            nn.Conv1d(in_channels='mask', out_channels='mask', kernel_size='mask', stride='mask', padding='mask'),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=0),
            nn.Conv1d(128, 256, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, 256, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.AvgPool1d(kernel_size=2, stride=2)
        )

        # Fully connected layers
        self.fc1 = nn.Linear(256, 512)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        # the initial shape of x is: (1, 1, 28)
        x = self.block1(x)  # output tensor shape: (1, 128, 3)
        x = x.flatten(1)  # output tensor shape: (1, 384)
        x = nn.Dropout(p=0.25)(x)  # output tensor shape: (1, 384)
        x = x.view(-1, 128)  # output tensor shape: (3, 128)
        x = self.block2(x)
        x = x.view(-1, 256)
        x = self.fc1(x)
        x = nn.ReLU(inplace=True)(x)
        x = self.fc2(x)
        return x

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 1, 28)  # Example input shape: (batch_size, channels, features)
    output_tensor = model(input_tensor)
    print(output_tensor.shape)