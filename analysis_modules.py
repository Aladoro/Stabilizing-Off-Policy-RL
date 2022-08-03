import torch.nn as nn

import utils
from drqv2 import Encoder
from analysis_layers import NonLearnableParameterizedRegWrapper, DummyParameterizedRegWrapper


class ReprRegularizedEncoder(Encoder):
    '''Encoder with regularization applied after final layer.'''
    def __init__(self, obs_shape, aug):
        nn.Module.__init__(self)
        assert len(obs_shape) == 3
        self.repr_dim = 32 * 35 * 35
        self.aug = aug
        self.convnet = nn.Sequential(nn.Conv2d(obs_shape[0], 32, 3, stride=2),
                                     nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
                                     nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
                                     nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
                                     nn.ReLU(), self.aug)

        self.apply(utils.weight_init)


class AllFeatRegularizedEncoder(Encoder):
    '''Encoder with different regularizations applied after every layer.'''
    def __init__(self, obs_shape, augs):
        nn.Module.__init__(self)
        self.augs = augs
        assert len(obs_shape) == 3
        assert len(augs) == 4

        self.repr_dim = 32 * 35 * 35
        layers = [nn.Conv2d(obs_shape[0], 32, 3, stride=2), nn.ReLU()]
        if self.augs[0] is not None:
            layers.append(self.augs[0])
        for i in range(3):
            layers += [nn.Conv2d(32, 32, 3, stride=1), nn.ReLU()]
            if self.augs[i+1] is not None:
                layers.append(self.augs[i+1])

        self.convnet = nn.Sequential(*layers)
        self.apply(utils.weight_init)


class AllFeatTiedRegularizedEncoder(ReprRegularizedEncoder):
    '''Encoder with the same regularization applied after every layer, and with the
       regularization parameter tuned only with the final layer's feature gradients.'''
    def __init__(self, obs_shape, aug):
        nn.Module.__init__(self)
        self.aug = aug
        assert len(obs_shape) == 3

        self.repr_dim = 32 * 35 * 35
        self.convnet = nn.Sequential(nn.Conv2d(obs_shape[0], 32, 3, stride=2),
                                     nn.ReLU(),
                                     NonLearnableParameterizedRegWrapper(self.aug),
                                     nn.Conv2d(32, 32, 3, stride=1),
                                     nn.ReLU(),
                                     NonLearnableParameterizedRegWrapper(self.aug),
                                     nn.Conv2d(32, 32, 3, stride=1),
                                     nn.ReLU(),
                                     NonLearnableParameterizedRegWrapper(self.aug),
                                     nn.Conv2d(32, 32, 3, stride=1),
                                     nn.ReLU(),
                                     self.aug)

        self.apply(utils.weight_init)


class LearnShiftRegularizedEncoder(ReprRegularizedEncoder):
    def __init__(self, obs_shape, aug):
        nn.Module.__init__(self)
        self.aug = aug
        assert len(obs_shape) == 3

        self.repr_dim = 32 * 35 * 35
        self.convnet = nn.Sequential(NonLearnableParameterizedRegWrapper(self.aug),
                                     nn.Conv2d(obs_shape[0], 32, 3, stride=2),
                                     nn.ReLU(),
                                     nn.Conv2d(32, 32, 3, stride=1),
                                     nn.ReLU(),
                                     nn.Conv2d(32, 32, 3, stride=1),
                                     nn.ReLU(),
                                     nn.Conv2d(32, 32, 3, stride=1),
                                     nn.ReLU(),
                                     DummyParameterizedRegWrapper(self.aug))

        self.apply(utils.weight_init)

