from torch import nn


class MultiLayerClassifier(nn.Module):
    def __init__(self, in_channels, out_classes):
        super(MultiLayerClassifier, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(in_channels, 128),
            nn.ReLU(),
            nn.Linear(128, out_classes),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)