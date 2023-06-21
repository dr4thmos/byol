import torchvision.models as models
import torch

class FinetuneEncoder(torch.nn.Module):
    def __init__(self, encoder, classifier, *args, **kwargs):
        super(FinetuneEncoder, self).__init__()
        
        self.encoder = encoder # resnet
        self.classifier = torch.nn.Sequential(
            classifier,
            torch.nn.Softmax()
        )
        
    def forward(self, x):
        h = self.encoder(x)
        print(h.shape)
        return self.classifier(h.view(h.shape[0], h.shape[1]))