from torch import nn


class LinearClassifier(nn.Module):
    def __init__(self, in_channels, out_classes):
        super(LinearClassifier, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(in_channels, out_classes),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)