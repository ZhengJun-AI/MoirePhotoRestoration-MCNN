from torch import nn

class MoireCNN(nn.Module):

    def conv_block(self, channels):
        convs = [nn.Conv2d(channels, channels, 3, 1, 1), nn.ReLU(True)] * 5
        return nn.Sequential(*convs)

    def up_conv_block(self, *channels):
        layer_nums = len(channels) - 1
        up_convs = []
        for num in range(layer_nums):
            up_convs += [nn.ConvTranspose2d(channels[num], channels[num + 1],
                                            4, 2, 1), nn.ReLU(True)]
        up_convs += [nn.Conv2d(32, 3, 3, 1, 1)]
        return nn.Sequential(*up_convs)

    def __init__(self):

        super().__init__()

        self.s11 = nn.Sequential(
            nn.Conv2d(3, 32, 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(32, 32, 3, 1, 1)
        )
        self.s12 = self.up_conv_block()
        self.s13 = self.conv_block(32)

        self.s21 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(32, 64, 3, 1, 1)
        )
        self.s22 = self.up_conv_block(64, 32)
        self.s23 = self.conv_block(64)

        init_conv = [nn.Conv2d(64, 64, 3, 2, 1), nn.ReLU(True),
                     nn.Conv2d(64, 64, 3, 1, 1)]
                     
        self.s31 = nn.Sequential(*init_conv)
        self.s32 = self.up_conv_block(64, 64, 32)
        self.s33 = self.conv_block(64)

        self.s41 = nn.Sequential(*init_conv)
        self.s42 = self.up_conv_block(64, 64, 32, 32)
        self.s43 = self.conv_block(64)

        self.s51 = nn.Sequential(*init_conv)
        self.s52 = self.up_conv_block(64, 64, 32, 32, 32)
        self.s53 = self.conv_block(64)

    def forward(self, x):
        x1 = self.s11(x)
        x2 = self.s21(x1)
        x3 = self.s31(x2)
        x4 = self.s41(x3)
        x5 = self.s51(x4)

        x1 = self.s12(self.s13(x1))
        x2 = self.s22(self.s23(x2))
        x3 = self.s32(self.s33(x3))
        x4 = self.s42(self.s43(x4))
        x5 = self.s52(self.s53(x5))

        x = x1 + x2 + x3 + x4 + x5

        return x
