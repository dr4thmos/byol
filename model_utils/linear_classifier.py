from torch import nn


class LinearClassifier(nn.Module):
    def __init__(self, in_channels, out_classes):
        super(MLPHead, self).__init__()

        nn.Linear(in_channels, out_classes)

    def forward(self, x):
        return self.net(x)