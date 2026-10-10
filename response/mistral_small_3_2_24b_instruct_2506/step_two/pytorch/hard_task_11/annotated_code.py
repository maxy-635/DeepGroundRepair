import torch
import torch.nn as nn

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Main pathway
        self.main_pathway = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(1, 1), stride=(1, 1)),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(num_features=64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )

        # Parallel branch
        self.parallel_branch = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(1, 1), stride=(1, 1)),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(num_features=64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1)),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(num_features=64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0)),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(num_features=64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )

        # Concatenation and 1x1 convolution
        self.concat_conv = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=(1, 1), stride=(1, 1)),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(num_features=64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )

        # Direct connection
        self.direct_connection = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(1, 1), stride=(1, 1)),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(num_features=64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )

        # Classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features='mask', out_features=512, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=512, out_features=10, bias=True),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        # the initial shape of x is: (1, 3, 32, 32)
        # Main pathway
        main_out = self.main_pathway(x)  # output tensor shape: (1, 64, 32, 32)

        # Parallel branch
        parallel_out = self.parallel_branch(x)  # output tensor shape: (1, 64, 32, 32)

        # Concatenate and process
        concat_out = torch.cat((main_out, parallel_out), dim=1)  # output tensor shape: (1, 128, 32, 32)
        concat_out = self.concat_conv(concat_out)  # output tensor shape: (1, 64, 32, 32)

        # Direct connection
        direct_out = self.direct_connection(x)  # output tensor shape: (1, 64, 32, 32)

        # Additive fusion
        fused_out = concat_out + direct_out  # output tensor shape: (1, 64, 32, 32)

        # Classifier
        out = self.classifier(fused_out)

        return out

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 3, 32, 32)  # Batch size of 1, 3 channels, 32x32 image
    output_tensor = model(input_tensor)
    print(model)