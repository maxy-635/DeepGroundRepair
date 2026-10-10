import torch
import torch.nn as nn
import torch.nn.functional as F

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Main pathway - 1x1 convolution
        self.main_path_1x1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True)
        )

        # Parallel branch - 1x1, 1x3, and 3x1 convolutions
        self.parallel_branch = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels=3, out_channels=64, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(num_features=64),
                nn.ReLU(inplace=True)
            ),
            nn.Sequential(
                nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(1, 3), stride=1, padding=(0, 1)),
                nn.BatchNorm2d(num_features=64),
                nn.ReLU(inplace=True)
            ),
            nn.Sequential(
                nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(3, 1), stride=1, padding=(1, 0)),
                nn.BatchNorm2d(num_features=64),
                nn.ReLU(inplace=True)
            )
        ])

        # Concatenation and final 1x1 convolution
        self.concat_conv = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=64, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True)
        )

        # Direct connection (skip connection)
        self.skip_connection = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(num_features=64)
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(in_features=64 * 32 * 32, out_features=512),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=512, out_features=10)
        )

    def forward(self, x):
        # Main pathway
        main_out = self.main_path_1x1(x)

        # Parallel branch
        parallel_outs = []
        for branch in self.parallel_branch:
            parallel_outs.append(branch(x))
        parallel_out = torch.cat(parallel_outs, dim=1)

        # Concatenate and process
        concat_out = self.concat_conv(parallel_out)

        # Add skip connection
        skip_out = self.skip_connection(x)
        out = concat_out + skip_out

        # Global average pooling
        out = F.adaptive_avg_pool2d(out, output_size=(1, 1))
        out = out.view(out.size(0), -1)

        # Classification
        out = self.classifier(out)

        return out

if __name__ == "__main__":
    model = DL_Model()
    # CIFAR-10 images are 32x32 with 3 channels
    input_tensor = torch.randn(1, 3, 32, 32)  # batch_size=1, channels=3, height=32, width=32
    output_tensor = model(input_tensor)
    print("Input shape:", input_tensor.shape)
    print("Output shape:", output_tensor.shape)