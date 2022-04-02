import torch
from torch.nn.modules import padding

class EncResNetBlock(torch.nn.Module):
    def __init__(self,
        in_channels,
        out_channels,
        stride
    ):
        super(EncResNetBlock, self).__init__()
        self.conv1 = torch.nn.Conv3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=(1, stride, stride),
            padding=1,
            bias=False
        )
        self.bn1 = torch.nn.BatchNorm3d(out_channels)
        self.conv2 = torch.nn.Conv3d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )
        self.bn2 = torch.nn.BatchNorm3d(out_channels)

        if stride == 1:
            self.shortcut = torch.nn.Sequential()
        else:
            self.shortcut = torch.nn.Sequential(
                torch.nn.Conv3d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=1,
                    stride=(1, stride, stride),
                    bias=False
                ),
                torch.nn.BatchNorm3d(out_channels)
            )

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = torch.relu(out)
        return out

class ResizeConv(torch.nn.Module):
    def __init__(self,
        in_channels,
        out_channels,
        kernel_size,
        scale_factor,
        mode = 'nearest'
    ):
        super(ResizeConv, self).__init__()
        self.conv = torch.nn.Conv3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=1
        )
        self.upscale = torch.nn.Upsample(scale_factor=(1, scale_factor, scale_factor), mode = mode)

    def forward(self, x):
        x = self.upscale(x)
        x = self.conv(x)
        return x

class DecResNetBlock(torch.nn.Module):
    def __init__(self,
        in_channels,
        out_channels,
        stride
    ):
        super(DecResNetBlock, self).__init__()
        self.conv2 = torch.nn.Conv3d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )
        self.bn2 = torch.nn.BatchNorm3d(in_channels)

        if stride == 1:
            self.conv1 = torch.nn.Conv3d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False
            )
            self.bn1 = torch.nn.BatchNorm3d(out_channels)
            self.shortcut = torch.nn.Sequential()

        else:
            self.conv1 = ResizeConv(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                scale_factor=stride
            )
            self.bn1 = torch.nn.BatchNorm3d(out_channels)
            self.shortcut = torch.nn.Sequential(
                ResizeConv(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=3,
                    scale_factor=stride
                ),
                torch.nn.BatchNorm3d(out_channels)
            )

    def forward(self, x):
        out = torch.relu(self.bn2(self.conv2(x)))
        out = self.bn1(self.conv1(out))
        out += self.shortcut(x)
        out = torch.relu(out)
        return out

class Autoencoder(torch.nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.conv1_1 = torch.nn.Conv3d(3, 16, kernel_size=3, stride=(1, 2, 2), padding=1, bias=False)
        self.conv1_2 = EncResNetBlock(16, 16, 1)
        self.conv1_3 = EncResNetBlock(16, 32, 2)

    
        self.conv2_1 = torch.nn.Conv3d(3, 16, kernel_size=3, stride=(1, 2, 2), padding=1, bias=False)
        self.conv2_2 = EncResNetBlock(16, 16, 1)
        self.conv2_3 = EncResNetBlock(16, 32, 2)

        self.conv4 = EncResNetBlock(64, 64, 2)
        self.conv5 = EncResNetBlock(64, 128, 2)

        self.conv6 = torch.nn.Conv3d(128, 1, kernel_size = 2, stride = (1, 1, 1), padding = (0, 0, 0))
        #self.conv7 = torch.nn.Conv3d(128, 256, kernel_size = 3, stride = (1, 2, 2), padding = (1, 0, 0))

        self.deconv5 = DecResNetBlock(128, 128, 2)
        self.deconv4 = DecResNetBlock(128, 64, 2)

        self.deconv3 = DecResNetBlock(64, 32, 2)
        self.deconv2 = DecResNetBlock(32, 32, 1)
        self.deconv1 = ResizeConv(32, 7, kernel_size=3, scale_factor=2)

    def forward(self, x):
        scale_1, scale_2 = torch.split(x, 3, 1)
        scale_1 = self.conv1_1(scale_1)
        scale_1 = self.conv1_2(scale_1)
        scale_1 = self.conv1_3(scale_1)
        scale_2 = self.conv2_1(scale_2)
        scale_2 = self.conv2_2(scale_2)
        scale_2 = self.conv2_3(scale_2)

        x = torch.cat([scale_1, scale_2], 1)
        x = self.conv4(x)
        z = self.conv5(x)

        traj = self.conv6(z)

        x = self.deconv5(z)
        x = self.deconv4(x)
       
        x = self.deconv3(x)
        x = self.deconv2(x) 
        x = self.deconv1(x)

        gen_image, depth = torch.split(x, [6, 1], 1)

        return z, [gen_image, depth, traj]
