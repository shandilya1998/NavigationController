import torch
from neurorobotics.constants import image_width, image_height

class ResizeConv2d(torch.nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, scale_factor, mode='nearest'):
        super(ResizeConv2d, self).__init__()
        self.scale_factor = scale_factor
        self.mode = mode
        self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=1)

    def forward(self, x):
        x = torch.nn.functional.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)
        x = self.conv(x)
        return x

class BasicBlockEnc(torch.nn.Module):

    def __init__(self, in_planes, stride=1, activation_fn = torch.nn.ReLU):
        super(BasicBlockEnc, self).__init__()

        planes = in_planes*stride

        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False),
            torch.nn.BatchNorm2d(planes),
            activation_fn()
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False),
            torch.nn.BatchNorm2d(planes),
        )

        if stride == 1:
            self.shortcut = torch.nn.Sequential()
        else:
            self.shortcut = torch.nn.Sequential(
                torch.nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                torch.nn.BatchNorm2d(planes)
            )
        self.activation = activation_fn()

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out += self.shortcut(x)
        out = self.activation(out)
        return out

class BasicBlockDec(torch.nn.Module):

    def __init__(self, in_planes, stride=1, activation_fn = torch.nn.ReLU):
        super(BasicBlockDec, self).__init__()

        planes = int(in_planes/stride)

        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(in_planes, in_planes, kernel_size=3, stride=1, padding=1, bias=False),
            torch.nn.BatchNorm2d(in_planes),
            activation_fn()
        )
        # self.bn1 could have been placed here, but that messes up the order of the layers when printing the class

        if stride == 1:
            self.conv1 = torch.nn.Sequential(
                torch.nn.Conv2d(in_planes, planes, kernel_size=3, stride=1, padding=1, bias=False),
                torch.nn.BatchNorm2d(planes),
                activation_fn()
            )
            self.shortcut = torch.nn.Sequential()
        else:
            self.conv1 = torch.nn.Sequential(
                ResizeConv2d(in_planes, planes, kernel_size=3, scale_factor=stride),
                torch.nn.BatchNorm2d(planes),
            )
            self.shortcut = torch.nn.Sequential(
                ResizeConv2d(in_planes, planes, kernel_size=3, scale_factor=stride),
                torch.nn.BatchNorm2d(planes)
            )
        self.activation = activation_fn()

    def forward(self, x):
        out = self.conv2(x)
        out = self.conv1(out)
        out += self.shortcut(x)
        out = self.activation(out)
        return out
"""
class ResNet18Enc(torch.nn.Module):

    def __init__(self, num_Blocks=[1,1,1,1], z_dim=10, nc=3):
        super(ResNet18Enc, self).__init__()
        self.in_planes = 64
        self.z_dim = z_dim
        self.conv1 = torch.nn.Conv2d(nc, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = torch.nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(BasicBlockEnc, 64, num_Blocks[0], stride=1)
        self.layer2 = self._make_layer(BasicBlockEnc, 128, num_Blocks[1], stride=2)
        self.layer3 = self._make_layer(BasicBlockEnc, 256, num_Blocks[2], stride=2)
        self.layer4 = self._make_layer(BasicBlockEnc, 512, num_Blocks[3], stride=2)
        self.linear = torch.nn.Linear(512, z_dim)

    def _make_layer(self, BasicBlockEnc, planes, num_Blocks, stride):
        strides = [stride] + [1]*(num_Blocks-1)
        layers = []
        for stride in strides:
            layers += [BasicBlockEnc(self.in_planes, stride)]
            self.in_planes = planes
        return torch.nn.Sequential(*layers)

    def forward(self, x):
        #print('Encoder')
        #print('input {}'.format(x.shape))
        x = torch.relu(self.bn1(self.conv1(x)))
        #print('conv1 {}'.format(x.shape))
        x = self.layer1(x)
        #print('layer1 {}'.format(x.shape))
        x = self.layer2(x)
        #print('layer2 {}'.format(x.shape))
        x = self.layer3(x)
        #print('layer3 {}'.format(x.shape))
        x = self.layer4(x)
        #print('layer4 {}'.format(x.shape))
        x = torch.nn.functional.adaptive_avg_pool2d(x, 1)
        #print('pool {}'.format(x.shape))
        x = x.view(x.size(0), -1)
        #print('reshape {}'.format(x.shape))
        x = self.linear(x)
        #print('output {}'.format(x.shape))
        return x

class ResNet18Dec(torch.nn.Module):

    def __init__(self, num_Blocks=[1,1,1,1], z_dim=10, nc=3):
        super(ResNet18Dec, self).__init__()
        self.in_planes = 512

        self.linear = torch.nn.Linear(z_dim, 512)
        self.nc = nc
        self.layer4 = self._make_layer(BasicBlockDec, 256, num_Blocks[3], stride=2)
        self.layer3 = self._make_layer(BasicBlockDec, 128, num_Blocks[2], stride=2)
        self.layer2 = self._make_layer(BasicBlockDec, 64, num_Blocks[1], stride=2)
        self.layer1 = self._make_layer(BasicBlockDec, 64, num_Blocks[0], stride=1)
        self.conv1 = ResizeConv2d(64, nc, kernel_size=3, scale_factor=2)
        self.output = torch.nn.Sigmoid()

    def _make_layer(self, BasicBlockDec, planes, num_Blocks, stride):
        strides = [stride] + [1]*(num_Blocks-1)
        layers = []
        for stride in reversed(strides):
            layers += [BasicBlockDec(self.in_planes, stride)]
        self.in_planes = planes
        return torch.nn.Sequential(*layers)

    def forward(self, z):
        #print('Decoder')
        #print('input {}'.format(z.shape))
        x = self.linear(z)
        #print('linear {}'.format(x.shape))
        x = x.view(z.size(0), 512, 1, 1)
        #print('reshape {}'.format(x.shape))
        x = torch.nn.functional.interpolate(x, scale_factor=4)
        #print('interpolate {}'.format(x.shape))
        x = self.layer4(x)
        #print('layer4 {}'.format(x.shape))
        x = self.layer3(x)
        #print('layer3 {}'.format(x.shape))
        x = self.layer2(x)
        #print('layer2 {}'.format(x.shape))
        x = self.layer1(x)
        #print('layer1 {}'.format(x.shape))
        x = self.output(self.conv1(x))
        #print('conv1 {}'.format(x.shape))
        x = x.view(x.size(0), self.nc, 64, 64)
        #print('output {}'.format(x.shape))
        return x
"""

class ResNet18Enc(torch.nn.Module):

    def __init__(self, num_Blocks=[1,1,1,1,1], nc=3, activation_fn = torch.nn.ReLU):
        super(ResNet18Enc, self).__init__()
        self.in_planes = 16
        self.conv1_1 = torch.nn.Sequential(
            torch.nn.Conv2d(nc, 16, kernel_size=5, stride=3, padding=2, bias=False),
            torch.nn.BatchNorm2d(16),
            activation_fn()
        )
        self.layer1_1 = self._make_layer(BasicBlockEnc, 16, num_Blocks[0], stride=1, activation_fn=activation_fn)
        self.layer2_1 = self._make_layer(BasicBlockEnc, 16, num_Blocks[1], stride=1, activation_fn=activation_fn)

        self.in_planes = 16
        self.conv1_2 = torch.nn.Sequential(
            torch.nn.Conv2d(nc, 16, kernel_size=5, stride=3, padding=2, bias=False),
            torch.nn.BatchNorm2d(16),
            activation_fn()
        )
        self.layer1_2 = self._make_layer(BasicBlockEnc, 16, num_Blocks[0], stride=1, activation_fn=activation_fn)
        self.layer2_2 = self._make_layer(BasicBlockEnc, 16, num_Blocks[1], stride=1, activation_fn=activation_fn)
  
        self.combiner = torch.nn.Sequential(
            torch.nn.Conv2d(16 * 2, 16, kernel_size = 3, padding = 1),
            activation_fn()
        )

    def _make_layer(self, BasicBlockEnc, planes, num_Blocks, stride, activation_fn):
        strides = [stride] + [1]*(num_Blocks-1)
        layers = []
        for stride in strides:
            layers += [BasicBlockEnc(self.in_planes, stride, activation_fn)]
            self.in_planes = planes
        return torch.nn.Sequential(*layers)

    def forward(self, x):
        scale_1, scale_2 = torch.split(x, 3, 1)
        scale_1 = self.conv1_1(scale_1)
        scale_1 = self.layer1_1(scale_1)
        scale_1 = self.layer2_1(scale_1)
        scale_2 = self.conv1_2(scale_2)
        scale_2 = self.layer1_2(scale_2)
        scale_2 = self.layer2_2(scale_2)

        x = torch.cat([scale_1, scale_2], 1)

        x = self.combiner(x)
        return x

class ResNet18Dec(torch.nn.Module):

    def __init__(self, num_Blocks=[1,1,1,1], nc=3, activation_fn = torch.nn.ReLU):
        super(ResNet18Dec, self).__init__()
        self.in_planes = 16

        self.nc = nc
        #self.layer6 = ResizeConv2d(2048, 1024, kernel_size=3, scale_factor=2)
        #self.layer5 = self._make_layer(BasicBlockDec, 512, num_Blocks[3], stride=2)
        self.layer2 = self._make_layer(BasicBlockDec, 5, num_Blocks[1], stride=3, activation_fn=activation_fn)
        self.layer1 = self._make_layer(BasicBlockDec, 5, num_Blocks[0], stride=1, activation_fn=activation_fn)
        self.conv1 = torch.nn.Conv2d(5, nc, kernel_size=3, stride=1, padding = 1)
        self.output = torch.nn.Sigmoid()


    def _make_layer(self, BasicBlockDec, planes, num_Blocks, stride, activation_fn):
        strides = [stride] + [1]*(num_Blocks-1)
        layers = []
        for stride in reversed(strides):
            layers += [BasicBlockDec(self.in_planes, stride, activation_fn)]
        self.in_planes = planes
        return torch.nn.Sequential(*layers)

    def forward(self, z):
        image = self.layer2(z)
        print('conv1', image.shape)
        image = self.layer1(image)
        print('conv2', image.shape)
        image = self.output(self.conv1(image))
        image = image.view(image.size(0), self.nc, image_width // 3, image_height // 3)
        return image

class Autoencoder(torch.nn.Module):

    def __init__(self, num_Blocks=[1,1,1,1], nc  = 3, activation_fn = torch.nn.ReLU):
        super(Autoencoder, self).__init__()
        self.encoder = ResNet18Enc(num_Blocks+[1], nc, activation_fn)
        self.decoder = ResNet18DecV2([2 * b for b in num_Blocks], nc, activation_fn)

    def forward(self, x):
        z = self.encoder(x)
        x = self.decoder(z)
        return z, x
