import torch

from collections import OrderedDict
from model.craft import CRAFT

# from model.FPN import FPN

import segmentation_models_pytorch as smp

class CraftMain():
    def __init__(self):
        self.model = CRAFT() # initialize

    def load_model(self, checkpoint, cuda=False):
        if cuda:
            self.model.load_state_dict(self.copyStateDict(torch.load(checkpoint)))
        else:
            self.model.load_state_dict(self.copyStateDict(torch.load(checkpoint, map_location='cpu')))

        return self.model

    def copyStateDict(self,state_dict):
        if list(state_dict.keys())[0].startswith("module"):
            start_idx = 1
        else:
            start_idx = 0
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = ".".join(k.split(".")[start_idx:])
            new_state_dict[name] = v
        return new_state_dict

class FPNMain():
    def __init__(self, backbone='resnext50', n_class=3):
        self.backbone = backbone
        self.n_class = n_class
        self.model =  FPN(encoder_name=self.backbone,
                decoder_pyramid_channels=256,
                decoder_segmentation_channels=128,
                classes=self.n_class,
                dropout=0.3,
                activation='sigmoid',
                final_upsampling=4,
                decoder_merge_policy='add')## Optimizer 설정

    def load_model(self, checkpoint, cuda=False):
        if cuda:
            state = torch.load(checkpoint)
        else:
            state = torch.load(checkpoint, map_location=torch.device('cpu'))
        self.model.load_state_dict(state['state_dict'])

        return self.model

# class SegmentationMain():
#     def __init__(self):
#         self.model = smp.FPN(encoder_name="resnext50_32x4d", classes=3)
#
#     def load_model(self, checkpoint, cuda=False):
#         if cuda:
#             state = torch.load(checkpoint)
#         else:
#             state = torch.load(checkpoint, map_location=torch.device('cpu'))
#         self.model.load_state_dict(state['model_state_dict'])