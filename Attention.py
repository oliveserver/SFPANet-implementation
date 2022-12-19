import time

import torch
from torch import nn


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=3):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel_size must be 3 or 7'
        padding = 1 if kernel_size == 3 else 3
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv1(out)
        out = self.sigmoid(out)
        return out


class CBAM(nn.Module):
    def __init__(self, channel, kernel_size=3):
        super(CBAM, self).__init__()
        assert kernel_size in (3, 7), 'kernel_size must be 3 or 7'
        self.ca = SeNet_ChannelAttention(channel)
        self.sa = SpatialAttention()

    def forward(self, x):
        shape = x.shape
        x = self.ca(x).reshape(shape[0], shape[1], 1, 1) * x
        x = self.sa(x) * x
        return x


class RowAttention(nn.Module):
    def __init__(self, in_dim, q_k_dim):
        super(RowAttention, self).__init__()
        self.in_dim = in_dim
        self.q_k_dim = q_k_dim

        self.q_conv = nn.Conv2d(in_dim, q_k_dim, 1, bias=False)
        self.k_conv = nn.Conv2d(in_dim, q_k_dim, 1, bias=False)
        self.v_conv = nn.Conv2d(in_dim, in_dim, 1, bias=False)
        self.softmax = nn.Softmax(dim=2)
        self.gama = nn.Parameter(torch.zeros(1), requires_grad=True)

    def forward(self, x):
        n, c, h, w = x.size()
        Q = self.q_conv(x)  # n,q_k_dim,h,w
        K = self.k_conv(x)  # n,q_k_dim,h,w
        V = self.v_conv(x)  # n,in_dim,h,w

        Q = Q.permute(0, 2, 1, 3).contiguous().view(n * h, -1, w).permute(0, 2, 1)  # n*h,w,q_k_dim
        K = K.permute(0, 2, 1, 3).contiguous().view(n * h, -1, w)  # n*h,q_k_dim,w
        V = V.permute(0, 2, 1, 3).contiguous().view(n * h, -1, w)  # n*h,in_dim,w

        row_att = torch.bmm(Q, K)  # n*h,w,w
        row_att = self.softmax(row_att)  # n*h,w,w

        out = torch.bmm(V, row_att.permute(0, 2, 1))  # n*h,in_dim,w
        out = out.view(n, h, -1, w).permute(0, 2, 1, 3)  # n,in_dim,h,w
        out = self.gama * out + x
        return out


class ColumnAttention(nn.Module):
    def __init__(self, in_dim, q_k_dim):
        super(ColumnAttention, self).__init__()
        self.in_dim = in_dim
        self.q_k_dim = q_k_dim
        self.q_conv = nn.Conv2d(in_dim, q_k_dim, 1, bias=False)
        self.k_conv = nn.Conv2d(in_dim, q_k_dim, 1, bias=False)
        self.v_conv = nn.Conv2d(in_dim, in_dim, 1, bias=False)
        self.softmax = nn.Softmax(dim=2)
        self.gama = nn.Parameter(torch.zeros(1), requires_grad=True)

    def forward(self, x):
        n, c, h, w = x.size()
        Q = self.q_conv(x)  # n,q_k_dim,h,w
        K = self.k_conv(x)  # n,q_k_dim,h,w
        V = self.v_conv(x)  # n,in_dim,h,w

        Q = Q.permute(0, 3, 1, 2).contiguous().view(n * w, -1, h).permute(0, 2, 1)  # n*h,w,q_k_dim
        K = K.permute(0, 3, 1, 2).contiguous().view(n * w, -1, h)  # n*h,q_k_dim,w
        V = V.permute(0, 3, 1, 2).contiguous().view(n * w, -1, h)  # n*h,in_dim,w

        row_att = torch.bmm(Q, K)  # n*h,w,w
        row_att = self.softmax(row_att)  # n*h,w,w

        out = torch.bmm(V, row_att.permute(0, 2, 1))  # n*h,in_dim,w
        out = out.view(n, w, -1, h).permute(0, 2, 3, 1)  # n,in_dim,h,w
        out = self.gama * out + x
        return out


class AA(nn.Module):
    def __init__(self, in_dim, q_k_dim=512, t=2):
        super(AA, self).__init__()
        self.list = [nn.Sequential(RowAttention(in_dim, q_k_dim), ColumnAttention(in_dim, q_k_dim)) for i in range(t)]
        self.layers = nn.ModuleList(self.list)

    def forward(self, x):
        for l in self.layers:
            x = l(x)
        return x


class SeNet_ChannelAttention(nn.Module):
    def __init__(self, channel, ratio=16):
        super(SeNet_ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.att = nn.Sequential(
            nn.Linear(channel, channel // ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // ratio, channel, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.avg_pool(x).squeeze()
        max_out = self.max_pool(x).squeeze()
        avg_out = self.att(avg_out)
        max_out = self.att(max_out)
        out = avg_out + max_out
        out = self.sigmoid(out)
        return out


class ChannelRowAttention(nn.Module):
    def __init__(self, in_dim, q_k_dim):
        super(ChannelRowAttention, self).__init__()
        self.in_dim = in_dim
        self.q_k_dim = q_k_dim
        self.q_conv = nn.Conv2d(in_dim, q_k_dim, 1, bias=False)
        self.k_conv = nn.Conv2d(in_dim, q_k_dim, 1, bias=False)
        self.v_conv = nn.Conv2d(in_dim, in_dim, 1, bias=False)
        self.softmax = nn.Softmax(dim=2)
        self.CA = SeNet_ChannelAttention(in_dim)
        self.gama = nn.Parameter(torch.zeros(1), requires_grad=True)

    def forward(self, x):
        n, c, h, w = x.size()
        Q = self.q_conv(x)  # n,q_k_dim,h,w
        K = self.k_conv(x)  # n,q_k_dim,h,w
        V = self.v_conv(x)  # n,in_dim,h,w

        Q = Q.permute(0, 2, 1, 3).reshape(n * h, -1, w).permute(0, 2, 1)  # n*h,w,q_k_dim
        K = K.permute(0, 2, 1, 3).reshape(n * h, -1, w)  # n*h,q_k_dim,w
        V = V.permute(0, 2, 1, 3).reshape(n * h, -1, w)  # n*h,in_dim,w

        row_att = torch.bmm(Q, K)  # n*h,w,w
        row_att = self.softmax(row_att)  # n*h,w,w

        out = torch.bmm(V, row_att.permute(0, 2, 1))  # n*h,in_dim,w
        out = out.view(n, h, -1, w).permute(0, 2, 1, 3)  # n,in_dim,h,w
        out = self.CA(out).reshape(n, c, 1, 1) * out
        out = self.gama * out + x
        return out


class ChannelColumnAttention(nn.Module):
    def __init__(self, in_dim, q_k_dim):
        super(ChannelColumnAttention, self).__init__()
        self.in_dim = in_dim
        self.q_k_dim = q_k_dim
        self.q_conv = nn.Conv2d(in_dim, q_k_dim, 1, bias=False)
        self.k_conv = nn.Conv2d(in_dim, q_k_dim, 1, bias=False)
        self.v_conv = nn.Conv2d(in_dim, in_dim, 1, bias=False)
        self.softmax = nn.Softmax(dim=2)
        self.CA = SeNet_ChannelAttention(in_dim)
        self.gama = nn.Parameter(torch.zeros(1), requires_grad=True)

    def forward(self, x):
        n, c, h, w = x.size()
        Q = self.q_conv(x)  # n,q_k_dim,h,w
        K = self.k_conv(x)  # n,q_k_dim,h,w
        V = self.v_conv(x)  # n,in_dim,h,w

        Q = Q.permute(0, 3, 1, 2).reshape(n * w, -1, h).permute(0, 2, 1)  # n*h,w,q_k_dim
        K = K.permute(0, 3, 1, 2).reshape(n * w, -1, h)  # n*h,q_k_dim,w
        V = V.permute(0, 3, 1, 2).reshape(n * w, -1, h)  # n*h,in_dim,w

        row_att = torch.bmm(Q, K)  # n*h,w,w
        row_att = self.softmax(row_att)  # n*h,w,w

        out = torch.bmm(V, row_att.permute(0, 2, 1))  # n*h,in_dim,w
        out = out.view(n, w, -1, h).permute(0, 2, 3, 1)  # n,in_dim,h,w
        out = self.CA(out).reshape(n, c, 1, 1) * out
        out = self.gama * out + x
        return out


# useless
class CAA(nn.Module):
    def __init__(self, in_dim, t=1):
        super(CAA, self).__init__()
        q_k_dim = in_dim // 8
        self.list = [nn.Sequential(ChannelRowAttention(in_dim, q_k_dim), ChannelColumnAttention(in_dim, q_k_dim)) for i
                     in range(t)]
        self.layers = nn.ModuleList(self.list)

    def forward(self, x):
        for l in self.layers:
            x = l(x)
        return x


# useless
class CCCA(nn.Module):
    def __init__(self, in_dim):
        super(CCCA, self).__init__()
        q_k_dim = 4
        self.q_conv = nn.Conv2d(in_dim, q_k_dim, 1, bias=False)
        self.k_conv = nn.Conv2d(in_dim, q_k_dim, 1, bias=False)
        self.v_conv = nn.Conv2d(in_dim, in_dim, 1, bias=False)
        self.softmax = nn.Softmax(dim=2)
        self.CA = SeNet_ChannelAttention(in_dim, ratio=1)
        self.gama = nn.Parameter(torch.zeros(1), requires_grad=True)

    def forward(self, x):
        n, c, h, w = x.size()
        Q = self.q_conv(x)  # n,q_k_dim,h,w
        K = self.k_conv(x)  # n,q_k_dim,h,w
        V = self.v_conv(x)  # n,in_dim,h,w

        Q_W = Q.permute(0, 2, 1, 3).reshape(n * h, -1, w).permute(0, 2, 1)  # n*h,w,q_k_dim
        K_W = K.permute(0, 2, 1, 3).reshape(n * h, -1, w)  # n*h,q_k_dim,w
        V_W = V.permute(0, 2, 1, 3).reshape(n * h, -1, w)  # n*h,in_dim,w

        row_att = torch.bmm(Q_W, K_W)  # n*h,w,w
        row_att = self.softmax(row_att)  # n*h,w,w
        Q_H = Q.permute(0, 3, 1, 2).reshape(n * w, -1, h).permute(0, 2, 1)  # n*w,h,q_k_dim
        K_H = K.permute(0, 3, 1, 2).reshape(n * w, -1, h)  # n*w,q_k_dim,h
        V_H = V.permute(0, 3, 1, 2).reshape(n * w, -1, h)  # n*w,in_dim,h

        col_att = torch.bmm(Q_H, K_H)  # n*w,h,h
        col_att = self.softmax(col_att)  # n*w,h,h

        out1 = torch.bmm(V_W, row_att.permute(0, 2, 1))  # n*h in_dim w
        out2 = torch.bmm(V_H, col_att.permute(0, 2, 1))  # n*w in_dim h
        out1 = out1.reshape(n, h, -1, w).permute(0, 2, 1, 3)
        out2 = out2.reshape(n, w, -1, h).permute(0, 2, 3, 1)
        out = out1 + out2
        out = self.CA(out).reshape(n, c, 1, 1) * out
        out = self.gama * out + x

        return out


class PositionAttention(nn.Module):
    def __init__(self, in_c):
        super(PositionAttention, self).__init__()
        self.q_k_dim = in_c // 8
        self.convq = nn.Conv2d(in_c, self.q_k_dim, 1, bias=False)
        self.convk = nn.Conv2d(in_c, self.q_k_dim, 1, bias=False)
        self.convv = nn.Conv2d(in_c, in_c, 1, bias=False)
        self.gama = nn.Parameter(torch.zeros(1), requires_grad=True)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, x):
        n, c, h, w = x.shape
        Q = self.convq(x).reshape(n, self.q_k_dim, h * w).permute(0, 2, 1)  # n h*w c
        K = self.convk(x).reshape(n, self.q_k_dim, h * w)  # n c h*w
        V = self.convv(x).reshape(n, c, h * w)  # n c h*w
        attention = self.softmax(torch.bmm(Q, K)).permute(0, 2, 1)  # n h*w h*w
        out = torch.bmm(V, attention).reshape(n, c, h, w)
        out = self.gama * out + x
        return out


class ChannelAttention(nn.Module):
    def __init__(self, in_c):
        super(ChannelAttention, self).__init__()
        self.q_k_dim = in_c
        self.convq = nn.Conv2d(in_c, self.q_k_dim, 1, bias=False)
        self.convk = nn.Conv2d(in_c, self.q_k_dim, 1, bias=False)
        self.convv = nn.Conv2d(in_c, in_c, 1, bias=False)
        self.gama = nn.Parameter(torch.zeros(1), requires_grad=True)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, x):
        n, c, h, w = x.shape
        Q = self.convq(x).reshape(n, self.q_k_dim, h * w)  # n c h*w
        K = self.convk(x).reshape(n, self.q_k_dim, h * w).permute(0, 2, 1)  # n h*w c
        V = self.convv(x).reshape(n, c, h * w).permute(0, 2, 1)  # n h*w c
        attention = self.softmax(torch.bmm(Q, K)).permute(0, 2, 1)  # n c c
        out = torch.bmm(V, attention).reshape(n, h, w, c).permute(0, 3, 1, 2)
        out = self.gama * out + x
        return out


class CrissCrossAttention(nn.Module):
    def __init__(self, in_dim):
        super(CrissCrossAttention, self).__init__()
        q_k_dim = in_dim // 8
        self.q_conv = nn.Conv2d(in_dim, q_k_dim, 1, bias=False)
        self.k_conv = nn.Conv2d(in_dim, q_k_dim, 1, bias=False)
        self.v_conv = nn.Conv2d(in_dim, in_dim, 1, bias=False)
        self.softmax = nn.Softmax(dim=2)
        self.gama = nn.Parameter(torch.zeros(1), requires_grad=True)

    def forward(self, x):
        n, c, h, w = x.size()
        Q = self.q_conv(x)  # n,q_k_dim,h,w
        K = self.k_conv(x)  # n,q_k_dim,h,w
        V = self.v_conv(x)  # n,in_dim,h,w

        Q_W = Q.permute(0, 2, 1, 3).reshape(n * h, -1, w).permute(0, 2, 1)  # n*h,w,q_k_dim
        K_W = K.permute(0, 2, 1, 3).reshape(n * h, -1, w)  # n*h,q_k_dim,w
        V_W = V.permute(0, 2, 1, 3).reshape(n * h, -1, w)  # n*h,in_dim,w

        row_att = torch.bmm(Q_W, K_W)  # n*h,w,w
        row_att = self.softmax(row_att)  # n*h,w,w

        Q_H = Q.permute(0, 3, 1, 2).reshape(n * w, -1, h).permute(0, 2, 1)  # n*w,h,q_k_dim
        K_H = K.permute(0, 3, 1, 2).reshape(n * w, -1, h)  # n*w,q_k_dim,h
        V_H = V.permute(0, 3, 1, 2).reshape(n * w, -1, h)  # n*w,in_dim,h

        col_att = torch.bmm(Q_H, K_H)  # n*w,h,h
        col_att = self.softmax(col_att)  # n*w,h,h

        out1 = torch.bmm(V_W, row_att.permute(0, 2, 1))  # n*h in_dim w
        out2 = torch.bmm(V_H, col_att.permute(0, 2, 1))  # n*w in_dim h
        out1 = out1.reshape(n, h, -1, w).permute(0, 2, 1, 3)
        out2 = out2.reshape(n, w, -1, h).permute(0, 2, 3, 1)
        out = out1 + out2
        out = self.gama * out + x
        return out


if __name__ == '__main__':
    x = torch.ones(10, 24, 20, 30)
    net = CrissCrossAttention(24)
    y = net(x)
    print(y.shape)
