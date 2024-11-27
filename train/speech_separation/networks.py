import torch
import torch.nn as nn
import torch.nn.functional as F


EPS = 1e-8


class network_wrapper(nn.Module):
    def __init__(self, args):
        super(network_wrapper, self).__init__()
        self.args = args
        if args.network == 'MossFormer2_SS_16K':
            from models.mossformer2.mossformer2 import MossFormer2_SS_16K
            self.ss_network = MossFormer2_SS_16K(args).model
        else:
            print("in networks, {args.network} is not found!")
            return

    def forward(self, mixture, visual=None):
        est_sources = self.ss_network(mixture)
        return est_sources
