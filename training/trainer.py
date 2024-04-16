
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader

import sys
sys.path.append("..")
from utils.utils import AverageMeter
from utils.utils import logger
from dataloader.pokemon_loader import PokemonDataSet
from dataloader import transforms
from tqdm import tqdm

import torchvision.models as models
import time

import ast



IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class Pokemon_Trainer(object):
    def __init__(self,args):
        super(Pokemon_Trainer, self).__init__()
        
        self.args = args
        
        self.lr = self.args.lr
        self.devices = self.args.devices
        self.devices = [int(item) for item in self.devices.split(',')]
        ngpu = len(self.devices)
        self.ngpu = ngpu
        
        
        self.datathread = self.args.datathread
        
        
        self.trainlist = self.args.trainlist
        self.vallist = self.args.vallist
        self.scale_size = self.args.scale_size
        
        self.datapath = self.args.datapath
        self.train_batch_size = self.args.train_batch_size
        self.val_batch_size = self.args.val_batch_size
        
        
        self.model_type = args.model_type
        
        self.initialize()
        
        
        
    def _build_optimizer(self):
        beta = 0.999
        momentum = 0.9
        self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.net.parameters()), self.lr,
                                        betas=(momentum, beta), amsgrad=True)


    def adjust_learning_rate(self, epoch):
        if epoch>=0 and epoch<=self.args.total_epochs//4:
            cur_lr = self.lr
        elif epoch>int(self.args.total_epochs//4) and epoch<=int(self.args.total_epochs//2):
            cur_lr = self.lr //4 *3
        elif epoch>int(self.args.total_epochs//2) and epoch<=int(self.args.total_epochs//4 *3):
            cur_lr = self.lr //2
        else:
            cur_lr = self.lr//4

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = cur_lr
        self.current_lr = cur_lr
        return cur_lr

    def set_criterion(self):
        self.criterion = nn.CrossEntropyLoss()

    
    def _build_net(self):
        if self.model_type=='resnet18':
            self.net = models.resnet18(pretrained=True)
            num_ftrs = self.net.fc.in_features
            class_num = 14 # 你的类别数量
            self.net.fc = nn.Linear(num_ftrs, class_num)
    
            # for name, param in self.net.named_parameters():
            #     if name.split(".")[0]!='fc':
            #         param.requires_grad = False
            #     else:
            #         param.requires_grad = True
                    
        
        self.net = torch.nn.DataParallel(self.net, device_ids=self.devices).cuda()
        
        
     
    def _prepare_dataset(self):
        
        train_transform_list = [   
                                transforms.ToTensor(),
                                    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)]
        
        train_transform = transforms.Compose(train_transform_list)
            
            
        val_transform_list = [transforms.ToTensor(),
                                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)]
            
        val_transform = transforms.Compose(val_transform_list)
            

        train_dataset = PokemonDataSet(datapath=self.datapath,
                                        trainlist=self.trainlist,
                                        vallist=self.vallist,
                                        mode='train',
                                        scale_size= [224,224],
                                        transforms = train_transform)

        test_dataset = PokemonDataSet(datapath=self.datapath,
                                        trainlist=self.trainlist,
                                        vallist=self.vallist,
                                        mode='val',
                                        scale_size=[224,224],
                                        transforms = val_transform)
            
            
        self.train_loader = DataLoader(train_dataset, batch_size = self.train_batch_size, \
                                    shuffle = True, num_workers = self.datathread, \
                                    pin_memory = True)

        self.test_loader = DataLoader(test_dataset, batch_size = self.val_batch_size, \
                                    shuffle = False, num_workers = self.datathread, \
                                    pin_memory = True)
        self.num_batches_per_epoch = len(self.train_loader)


    def initialize(self):
        self._prepare_dataset()
        self._build_net()
        self._build_optimizer()
        
    def train_one_epoch(self, epoch, round,iterations):

        # Data Summary
        batch_time = AverageMeter()
        data_time = AverageMeter()    
        losses = AverageMeter()
        
        acc_rate_meter = AverageMeter()
        

        nums_samples = len(self.train_loader)
        train_count = 0
        
        # # non-detection
        # torch.autograd.set_detect_anomaly(True)
        
        # switch to train mode
        self.net.train()
        end = time.time()
        cur_lr = self.adjust_learning_rate(epoch)
        logger.info("learning rate of epoch %d: %f." % (epoch, cur_lr))
        
        batch_id = 0
        for sample_batched in tqdm(self.train_loader):
            
            batch_id = batch_id+1
            
            image_data = torch.autograd.Variable(sample_batched['img'].cuda(), requires_grad=False)
            labels = torch.autograd.Variable(sample_batched['label'].cuda(), requires_grad=False)
            
            data_time.update(time.time() - end)
            self.optimizer.zero_grad()
            
            
            if self.model_type=='resnet18':
                outputs = self.net(image_data)
                loss = self.criterion(outputs,labels)
                
     
            # compute gradient and do SGD step
            with torch.autograd.detect_anomaly():
                loss.backward()
                
            self.optimizer.step()
            iterations = iterations+1
            
            
            batch_time.update(time.time() - end)
            end = time.time()
            
            losses.update(loss.data.item(), image_data.size(0))

            val_correct = 0
            val_total = 0
            _, val_predicted = torch.max(outputs.data, 1)
            val_total += labels.size(0)
            val_correct += (val_predicted == labels).sum().item()
            acc_rate = val_correct*1.0 /val_total
            
            acc_rate_meter.update(acc_rate,1)

            if batch_id%5==0:
                logger.info("Epoch {}\t, Loss : {} \t, Acc: {}\t".format(epoch,losses.val,acc_rate_meter.val))
                
        
        return losses.avg,acc_rate_meter.val , iterations
            
            

    def validate(self,epoch):
        
        batch_time = AverageMeter()
        acc_meter = AverageMeter()
        
        self.net.eval()
        end = time.time()
        inference_time = 0
        img_nums = 0
        nums_samples = len(self.test_loader)
        test_count = 0
        
        val_correct = 0
        val_total = 0
        
        for sample_batched in tqdm(self.test_loader):
            
            img = torch.autograd.Variable(sample_batched['img'].cuda(), requires_grad=False)
            labels = torch.autograd.Variable(sample_batched['label'].cuda(), requires_grad=False)
            
            
            # INFEREMCE            
            with torch.no_grad():
                outputs = self.net(img)
                _, val_predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (val_predicted == labels).sum().item()
                
                
                

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            
            
        
        val_acc_cur_epoch = 100 * val_correct/val_total
        
        
        
        logger.info("Epoch {}'s accurate rate is {}".format(epoch,val_acc_cur_epoch))
        
        
        return val_acc_cur_epoch

    def get_model(self):
        return self.net.state_dict()    
            
            
        
        
        
        
        
        pass




