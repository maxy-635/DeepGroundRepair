import paddle
import paddle.nn as nn

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()
        # Encoder part: 3 pairs of <convolutional layer, max-pooling layer>
        self.encoder_conv1 = nn.Sequential(
            nn.Conv2D(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2D(kernel_size=2, stride=2, padding=0)
        )
        self.encoder_conv2 = nn.Sequential(
            nn.Conv2D(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2D(kernel_size=2, stride=2, padding=0)
        )
        self.encoder_conv3 = nn.Sequential(
            nn.Conv2D(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2D(kernel_size=2, stride=2, padding=0)
        )

        # Middle part: <convolutional layer, Dropout layer, convolutional layer>
        self.middle_conv1 = nn.Sequential(
            nn.Conv2D(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.Dropout(p=0.2),
            nn.Conv2D(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        )

        # Decoder part: 3 pairs of <convolutional layer, transposed convolutional layer> with skip connections
        self.decoder_conv1 = nn.Sequential(
            nn.Conv2D(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.Conv2DTranspose(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1)
        )
        self.decoder_conv2 = nn.Sequential(
            nn.Conv2D(in_channels=256, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.Conv2DTranspose(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1)
        )
        self.decoder_conv3 = nn.Sequential(
            nn.Conv2D(in_channels=128, out_channels=3, kernel_size=3, stride=1, padding=1),
            nn.Conv2DTranspose(in_channels=3, out_channels=10, kernel_size=4, stride=2, padding=1)
        )

        # Skip connections should be adjusted to match the decoder's spatial dimensions
        self.skip_connection1 = nn.Sequential(
            nn.Conv2D(in_channels=256, out_channels=256, kernel_size=1, stride=2, padding=0)  # Added stride=2 to match upsampling
        )
        self.skip_connection2 = nn.Sequential(
            nn.Conv2D(in_channels=128, out_channels=128, kernel_size=1, stride=2, padding=0)  # Added stride=2 to match upsampling
        )
        self.skip_connection3 = nn.Sequential(
            nn.Conv2D(in_channels=64, out_channels=64, kernel_size=1, stride=2, padding=0)    # Added stride=2 to match upsampling
        )

    def forward(self, x):
        # Encoder
        x1 = self.encoder_conv1(x)
        x2 = self.encoder_conv2(x1)
        x3 = self.encoder_conv3(x2)

        # Middle part
        x = self.middle_conv1(x3)

        # Decoder with skip connections
        x = self.skip_connection1(x3) + self.decoder_conv1(x)
        x = self.skip_connection2(x2) + self.decoder_conv2(x)
        x = self.skip_connection3(x1) + self.decoder_conv3(x)

        # Flatten and classify
        x = nn.Flatten()(x)
        x = nn.Linear(in_features=256, out_features=10)(x)
        return x