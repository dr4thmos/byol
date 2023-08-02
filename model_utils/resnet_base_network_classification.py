import torchvision.models as models
import torch
from model_utils.mlp_head import MLPHead


class ResNet(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super(ResNet, self).__init__()
        if kwargs['name'] == 'resnet18':
            if (kwargs["pretrained_weights"] == True) & (kwargs['fine_tune_from'] is None):
                resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
                print("Resnet18 model with imagenet weight loaded")
            else:
                resnet = models.resnet18(weights=None)
                print("Resnet18 model with random weight loaded")
        elif kwargs['name'] == 'resnet50':
            if (kwargs["pretrained_weights"] == True) & (kwargs['fine_tune_from'] is None):
                resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
                print("Resnet50 model with imagenet weight loaded")
            else:
                resnet = models.resnet50(weights=None)
                print("Resnet50 model with random weight loaded")
                
        self.repr_shape = resnet.fc.in_features
        self.encoder = torch.nn.Sequential(*list(resnet.children())[:-1])
        #self.encoder.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        
        self.projetion = MLPHead(in_channels=resnet.fc.in_features, **kwargs['projection_head'])
        

    def forward(self, x, repr=False):
        h = self.encoder(x)
        if repr:
            return h.view(h.shape[0], h.shape[1])
        else:
            h = h.view(h.shape[0], h.shape[1])
            return self.projetion(h)