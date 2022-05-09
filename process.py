import torch

from collections import OrderedDict
from model.craft import CRAFT

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

