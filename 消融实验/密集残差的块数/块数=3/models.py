from torch import nn
import torch
import numpy as np
import os
from math import sqrt

def conv_block(in_channel, out_channel):
    layer = nn.Sequential(
        nn.ReLU(),
        nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1, bias=False)
    )
    return layer

class dense_block(nn.Module):
    def __init__(self, in_channel, growth_rate,out_channel, num_layers):
        super(dense_block, self).__init__()
        block = []
        channel = in_channel
        for i in range(num_layers):
            block.append(conv_block(channel, growth_rate))
            channel += growth_rate
        self.net = nn.Sequential(*block)
        self.conv1=nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels=(num_layers)*growth_rate+in_channel,out_channels=out_channel,kernel_size=1,bias=False)
        )

    def forward(self, x):
        for layer in self.net:
            out = layer(x)
            x = torch.cat((out, x), dim=1)
        x=self.conv1(x)
        return x

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.residual_layer=dense_block(in_channel=64,growth_rate=12,out_channel=64,num_layers=3)
        self.input = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.output = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.ReLU()
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, sqrt(2. / n))

    def make_layer(self, block, num_of_layer):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block())
        return nn.Sequential(*layers)

    def forward(self, x):
        residual = x
        out = self.relu(self.input(x))
        out = self.residual_layer(out)
        out = self.output(out)
        out = torch.add(out, residual)
        return out
if __name__=="__main__":
    import torch
    from torchinfo import summary
    from torchstat import stat
    x=torch.rand(1,1,48,48)
    model=Net()
    out=model(x)
    # summary(model,x.shape)
    x=torch.rand(1,48,48)
    stat(model,x.shape)
