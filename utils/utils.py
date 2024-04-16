import json
import yaml
import logging
import os
import numpy as np
import sys

def load_loss_scheme(loss_config):

    with open(loss_config, 'r') as f:
        loss_json = yaml.safe_load(f)

    return loss_json

DEBUG =0
logger = logging.getLogger()

if DEBUG:
    #coloredlogs.install(level='DEBUG')
    logger.setLevel(logging.DEBUG)
else:
    #coloredlogs.install(level='INFO')
    logger.setLevel(logging.INFO)

strhdlr = logging.StreamHandler()
logger.addHandler(strhdlr)
formatter = logging.Formatter('%(asctime)s [%(filename)s:%(lineno)d] %(levelname)s %(message)s')
strhdlr.setFormatter(formatter)




class AverageMeter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
