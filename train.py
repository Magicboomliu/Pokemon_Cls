from __future__ import print_function
import os
import argparse
import datetime
import random
import torch
import logging
import torch.nn as nn
import torch.backends.cudnn as cudnn
from utils.utils import *
from training.trainer import Pokemon_Trainer



from torch.utils.tensorboard import SummaryWriter
cudnn.benchmark = True


def save_checkpoint(state, is_best, filename='checkpoint.pth'):
    torch.save(state, os.path.join(opt.outf,filename))
    if is_best:
        torch.save(state, os.path.join(opt.outf,'model_best.pth'))

'''  Main Function to train the model'''
def main(opt):
    
    # initialize a trainer
    trainer = Pokemon_Trainer(args=opt)
    
    # validate the pretrained model on test data
    best_acc = -1
    best_index = 0
    start_epoch = opt.startEpoch 

    # if trainer.is_pretrain:
    #     pass
    #     # best_EPE = trainer.validate(summary_writer=summary_writer,epoch=start_epoch)

    iterations = 0
    
    for r in range(0, 1):
        
        # Set your loss here
        trainer.set_criterion()
    
        logger.info('num of epoches: %d' % opt.total_epochs)
        logger.info('\t'.join(['epoch', 'time_stamp', 'train_loss', 'train_EPE', 'EPE', 'lr']))
        
        for i in range(start_epoch, opt.total_epochs):
            avg_loss,avg_acc,iterations = trainer.train_one_epoch(i, r,iterations)
            val_acc = trainer.validate(i)
            is_best = best_acc < 0 or val_acc > best_acc

            if is_best:
                best_acc = val_acc
                best_index = i
         
            save_checkpoint({
                    'round': r + 1,
                    'epoch': i + 1,
                    'arch': 'dispnet',
                    'state_dict': trainer.get_model(),
                    'best_Acc': best_acc,
                }, is_best, '%s_%d_%d_%.3f.pth' % (opt.model_type, r, i, val_acc))
        
            #logger.info('Validation[epoch:%d]: '%i+'\t'.join([datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), str(avg_loss), str(a), str(val_EPE), str(trainer.current_lr)]))
            logger.info("Best Acc from %d epoch" % (best_index))
        
        start_epoch = 0



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--model_type', type=str, help='logging file', default='./train.log')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate, default=0.0002')
    parser.add_argument('--devices', type=str, help='indicates CUDA devices, e.g. 0,1,2', default='0')
    parser.add_argument('--datapath', type=str, help='provide the root path of the data', default='/spyder/sceneflow/')
    parser.add_argument('--datathread', type=int, help='provide the root path of the data', default='/spyder/sceneflow/')
    parser.add_argument('--trainlist', type=str, help='provide the root path of the data', default='/spyder/sceneflow/')
    parser.add_argument('--vallist', type=str, help='provide the root path of the data', default='/spyder/sceneflow/')
    parser.add_argument('--train_batch_size', type=int, help='provide the root path of the data', default='/spyder/sceneflow/')
    parser.add_argument('--val_batch_size', type=int, help='provide the root path of the data', default='/spyder/sceneflow/')
    parser.add_argument('--scale_size', type=str, help='provide the root path of the data', default='[224,224]')
    

    
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum for sgd, alpha parameter for adam. default=0.9')
    parser.add_argument('--beta', type=float, default=0.999, help='beta parameter for adam. default=0.999')
    parser.add_argument('--cuda', action='store_true', help='enables, cuda')
    
    parser.add_argument('--outf', default='.', help='folder to output images and model checkpoints')
    parser.add_argument('--manualSeed', type=int, help='manual seed')
    

    parser.add_argument('--startEpoch', type=int, help='the epoch number to start training, useful of lr scheduler', default='0')
    parser.add_argument('--logFile', type=str, help='logging file', default='./train.log')
    parser.add_argument('--showFreq', type=int, help='display frequency', default='100')
    parser.add_argument('--flowDiv', type=float, help='the number by which the flow is divided.', default='1.0')
    
    parser.add_argument('--total_epochs', type=int, help='provide the root path of the data', default='/spyder/sceneflow/')



    opt = parser.parse_args()

    try:
        os.makedirs(opt.outf)
    except OSError:
        pass
    
    os.makedirs('logs',exist_ok=True)
    hdlr = logging.FileHandler(opt.logFile)
    hdlr.setFormatter(formatter)
    logger.addHandler(hdlr) 
    logger.info('Configurations: %s', opt)
    
    if opt.manualSeed is None:
        opt.manualSeed = random.randint(1, 10000)
        
    logger.info("Random Seed: %s", opt.manualSeed)
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)
    if opt.cuda:
        torch.cuda.manual_seed_all(opt.manualSeed)

    if torch.cuda.is_available() and not opt.cuda:
        logger.warning("WARNING: You should run with --cuda since you have a CUDA device.")
    main(opt)