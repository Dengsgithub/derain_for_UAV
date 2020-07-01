import os
import sys
import cv2
import argparse
import numpy as np
import itertools
import time
import torch
from torch import nn
from torch.nn import DataParallel
from torch.optim import Adam
from torch.autograd import Variable 
from torch.utils.data import DataLoader
from collections import Counter
from dataset import ShowDataset
# from model import RESCAN
from unet.unet_model import UNet
logger = settings.logger
torch.cuda.manual_seed_all(66)
torch.manual_seed(66)
torch.cuda.set_device(settings.device_id)


def ensure_dir(dir_path):
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)
        

class Session:
    def __init__(self):
        self.show_dir = "../showdir_dark_train"
        self.model_dir = "../models_dark"
        ensure_dir(settings.show_dir)
        ensure_dir(settings.model_dir)
        logger.info('set show dir as %s' % "../showdir_dark_train")
        logger.info('set model dir as %s' % "../models_dark")

        self.net = UNet(3, 3).cuda()
        self.dataset = None 
        self.dataloader = None 

    def get_dataloader(self, dataset_name):
        self.dataset = ShowDataset(dataset_name)
        self.dataloader = \
                    DataLoader(self.dataset, batch_size=1, 
                            shuffle=False, num_workers=1)
        return self.dataloader

    def load_checkpoints(self, name):
        ckp_path = os.path.join(self.model_dir, name)
        try:
            obj = torch.load(ckp_path)
            logger.info('Load checkpoint %s' % ckp_path)
        except FileNotFoundError:
            logger.info('No checkpoint %s!!' % ckp_path)
            return
        self.net.load_state_dict(obj['net'])

    def inf_batch(self, name, batch):
        O = batch['O'].cuda()
        O = Variable(O, requires_grad=False)

        with torch.no_grad():
            derain = self.net(O)

        return derain

    def save_image(self, No, imgs):
        for i, img in enumerate(imgs):
            img = (img.cpu().data * 255).numpy()
            img = np.clip(img, 0, 255)
            img = np.transpose(img, (1, 2, 0))
            # h, w, c = img.shape
            # if i == 3:
            img_file = os.path.join(self.show_dir, '%s.png' % (No))
            cv2.imread(os.path.join(
                "D:\\Desktop\\Code\\pytorch\\RESCAN-master\\dataset\\c\\Rain_200_H\\test", '%s.png' % (No)))
            cv2.imwrite(img_file, img)


def run_show(ckp_name):
    sess = Session()
    sess.load_checkpoints(ckp_name)
    sess.net.eval()

    dt = sess.get_dataloader('test')
    all_time =[]
    for i, batch in enumerate(dt):
        logger.info(i)
        start_time = time.time()
        imgs = sess.inf_batch('test', batch)
        stop_time =time.time()
        all_time.append(stop_time-start_time)
        No = sess.dataset.get_name(batch['idx'][0])
        sess.save_image(No, imgs)
    print(np.mean(all_time))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', default='latest')

    args = parser.parse_args(sys.argv[1:])
    run_show("epoch 140_ssim 0.769225")

