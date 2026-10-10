# class Block1(paddle.nn.Layer):
#     def __init__(self, in_channels):
#         super().__init__()
#         self.conv1 = paddle.nn.Conv2D(in_channels, in_channels//3, kernel_size=1, padding=0)
#         self.conv2 = paddle.nn.Conv2D(in_channels, in_channels//3, kernel_size=1, padding=0)
#         self.conv3 = paddle.nn.Conv2D(in_channels, in_channels//3, kernel_size=1, padding=0)
#         self.relu = paddle.nn.ReLU()

#     def forward(self, x):
#         x1 = self.conv1(x)
#         x2 = self.conv2(x)
#         x3 = self.conv3(x)
#         x = paddle.concat([x1, x2, x3], axis=1)
#         x = self.relu(x)
#         return x

# class Block2(paddle.nn.Layer):
#     def __init__(self, in_channels):
#         super().__init__()

#     def forward(self, x):
#         # Reshape the feature map to (height, width, groups, channels_per_group)
#         # groups = 3
#         # channels_per_group = in_channels // 3
#         # Reshape the feature map to (height, width, groups, channels_per_group)
#         # Swap the third and fourth dimensions
#         # Reshape the feature map back to its original shape
#         # Return the reshaped feature map
#         # Your code here
#         pass

# class Block3(paddle.nn.Layer):
#     def __init__(self, in_channels):
#         super().__init__()
#         self.depthwise_conv = paddle.nn.Conv2D(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels)

#     def forward(self, x):
#         x = self.depthwise_conv(x)
#         return x