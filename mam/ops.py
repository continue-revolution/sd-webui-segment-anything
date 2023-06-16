import torch
from   torch import nn
from   torch.nn import Parameter


def l2normalize(v, eps=1e-12):
    return v / (v.norm() + eps)


class SpectralNorm(nn.Module):
    """
    Based on https://github.com/heykeetae/Self-Attention-GAN/blob/master/spectral.py
    and add _noupdate_u_v() for evaluation
    """
    def __init__(self, module, name='weight', power_iterations=1):
        super(SpectralNorm, self).__init__()
        self.module = module
        self.name = name
        self.power_iterations = power_iterations
        if not self._made_params():
            self._make_params()

    def _update_u_v(self):
        u = getattr(self.module, self.name + "_u")
        v = getattr(self.module, self.name + "_v")
        w = getattr(self.module, self.name + "_bar")

        height = w.data.shape[0]
        for _ in range(self.power_iterations):
            v.data = l2normalize(torch.mv(torch.t(w.view(height,-1).data), u.data))
            u.data = l2normalize(torch.mv(w.view(height,-1).data, v.data))

        sigma = u.dot(w.view(height, -1).mv(v))
        setattr(self.module, self.name, w / sigma.expand_as(w))

    def _noupdate_u_v(self):
        u = getattr(self.module, self.name + "_u")
        v = getattr(self.module, self.name + "_v")
        w = getattr(self.module, self.name + "_bar")

        height = w.data.shape[0]
        sigma = u.dot(w.view(height, -1).mv(v))
        setattr(self.module, self.name, w / sigma.expand_as(w))

    def _made_params(self):
        try:
            u = getattr(self.module, self.name + "_u")
            v = getattr(self.module, self.name + "_v")
            w = getattr(self.module, self.name + "_bar")
            return True
        except AttributeError:
            return False

    def _make_params(self):
        w = getattr(self.module, self.name)

        height = w.data.shape[0]
        width = w.view(height, -1).data.shape[1]

        u = Parameter(w.data.new(height).normal_(0, 1), requires_grad=False)
        v = Parameter(w.data.new(width).normal_(0, 1), requires_grad=False)
        u.data = l2normalize(u.data)
        v.data = l2normalize(v.data)
        w_bar = Parameter(w.data)

        del self.module._parameters[self.name]

        self.module.register_parameter(self.name + "_u", u)
        self.module.register_parameter(self.name + "_v", v)
        self.module.register_parameter(self.name + "_bar", w_bar)

    def forward(self, *args):
        # if torch.is_grad_enabled() and self.module.training:
        if self.module.training:
            self._update_u_v()
        else:
            self._noupdate_u_v()
        return self.module.forward(*args)


class ASPP(nn.Module):
    '''
    based on https://github.com/chenxi116/DeepLabv3.pytorch/blob/master/deeplab.py
    '''
    def __init__(self, in_channel, out_channel, conv=nn.Conv2d, norm=nn.BatchNorm2d):
        super(ASPP, self).__init__()
        mid_channel = 256
        dilations = [1, 2, 4, 8]

        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.relu = nn.ReLU(inplace=True)
        self.aspp1 = conv(in_channel, mid_channel, kernel_size=1, stride=1, dilation=dilations[0], bias=False)
        self.aspp2 = conv(in_channel, mid_channel, kernel_size=3, stride=1,
                               dilation=dilations[1], padding=dilations[1],
                               bias=False)
        self.aspp3 = conv(in_channel, mid_channel, kernel_size=3, stride=1,
                               dilation=dilations[2], padding=dilations[2],
                               bias=False)
        self.aspp4 = conv(in_channel, mid_channel, kernel_size=3, stride=1,
                               dilation=dilations[3], padding=dilations[3],
                               bias=False)
        self.aspp5 = conv(in_channel, mid_channel, kernel_size=1, stride=1, bias=False)
        self.aspp1_bn = norm(mid_channel)
        self.aspp2_bn = norm(mid_channel)
        self.aspp3_bn = norm(mid_channel)
        self.aspp4_bn = norm(mid_channel)
        self.aspp5_bn = norm(mid_channel)
        self.conv2 = conv(mid_channel * 5, out_channel, kernel_size=1, stride=1,
                               bias=False)
        self.bn2 = norm(out_channel)

    def forward(self, x):
        x1 = self.aspp1(x)
        x1 = self.aspp1_bn(x1)
        x1 = self.relu(x1)
        x2 = self.aspp2(x)
        x2 = self.aspp2_bn(x2)
        x2 = self.relu(x2)
        x3 = self.aspp3(x)
        x3 = self.aspp3_bn(x3)
        x3 = self.relu(x3)
        x4 = self.aspp4(x)
        x4 = self.aspp4_bn(x4)
        x4 = self.relu(x4)
        x5 = self.global_pooling(x)
        x5 = self.aspp5(x5)
        x5 = self.aspp5_bn(x5)
        x5 = self.relu(x5)
        x5 = nn.Upsample((x.shape[2], x.shape[3]), mode='nearest')(x5)
        x = torch.cat((x1, x2, x3, x4, x5), 1)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        return x