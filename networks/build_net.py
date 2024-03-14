import logging
from .CVAE import *
from .CVAE32 import *
from .Discriminator import *
from .EFnet import *


def build_CVAD_net(dataset_name, net_name, capacity, channel):   
    
    logger = logging.getLogger()
    logger.info("Build CVAD embnet for {}".format(dataset_name))
    assert dataset_name in ['cifar10', 'siim', 'coco']
    
    if dataset_name == "cifar10":
        netG = CVAE32(capacity,channel)
        netD = Discriminator32(capacity, channel)
        
    else:
        netG = CVAE(capacity,channel)
        netD = Discriminator(capacity, channel)
        
    return netG, netD

def build_EF_net():
    
    efnet = EarlyFusionNetwork()
    return efnet
