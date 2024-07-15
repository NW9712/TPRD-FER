import torch.nn
import torch.nn as nn

class CombinedAttn(nn.Module):

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, kernel_size: list, stride: list, padding: list, dilation: list):
        super(CombinedAttn, self).__init__()
        self.conv1x1 = nn.Sequential(
            nn.Conv2d(input_dim, hidden_dim, kernel_size=(1,1)),
            nn.BatchNorm2d(hidden_dim ),
        )

        self.conv_nxn = nn.Sequential(
            nn.Conv2d(hidden_dim, output_dim, kernel_size=(kernel_size[0],kernel_size[0]), stride=(stride[0],stride[0]), padding=(padding[0],padding[0]), dilation=(dilation[0],dilation[0])),
            nn.BatchNorm2d(output_dim),
        )
        self.conv_sxn = nn.Sequential(
            nn.Conv2d(hidden_dim, output_dim, kernel_size=(stride[1],kernel_size[1]), stride=(stride[1],stride[1]), padding=(0,padding[1]), dilation=(dilation[1],dilation[1])),
            nn.BatchNorm2d(output_dim),
        )
        self.conv_nxs = nn.Sequential(
            nn.Conv2d(hidden_dim, output_dim, kernel_size=(kernel_size[2],stride[2]), stride=(stride[2],stride[2]), padding=(padding[2],0), dilation=(dilation[2],dilation[2])),
            nn.BatchNorm2d(output_dim),
        )
        self.prelu = nn.RReLU()

    def forward(self, feature_map_in, feature_map_out):
        y = self.conv1x1(feature_map_in)
        y = self.prelu(self.conv_nxn(y) + self.conv_sxn(y) + self.conv_nxs(y))
        y = y.sum(dim=1, keepdim=True)
        out = feature_map_out  * y

        return out

class CBAM_SpatialAttention(nn.Module):
    def __init__(self, kernel_size):
        super(CBAM_SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ =torch.max(x, dim=1, keepdim=True)
        y = torch.cat([avg_out, max_out], dim = 1)
        y = self.conv1(y)
        return x * self.sigmoid(y)

class SE_ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16, pool = ['avg', 'max']):
        super(SE_ChannelAttention, self).__init__()
        assert 'avg' in pool or 'max' in pool
        self.pool = pool
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
                                nn.PReLU(),
                                nn.Conv2d(in_planes//ratio, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = 0
        if 'avg' in self.pool:
            avg_out = self.fc(self.avg_pool(x))
            y += avg_out
        if 'max' in self.pool:
            max_out = self.fc(self.max_pool(x))
            y += max_out
        return x * self.sigmoid(y)