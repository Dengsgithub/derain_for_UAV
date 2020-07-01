import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import sys
import cv2
import argparse
import numpy as np
import logging

import torch
from torch import nn
from torch.nn import MSELoss
from torch.nn import L1Loss
from Myloss import Myloss
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid

from ERL import ERLNet
from unet.unet_model import UNet
from ERL import ERL_baseline
from dataset import TrainDataset, TestDataset
from cal_ssim import SSIM


def get_args():
    parser = argparse.ArgumentParser(description="train derain model",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--train_dir", type=str, default='../dataset/c/Rain_200_H/train',
                        help="train image dir")
    parser.add_argument("--test_dir", type=str, default='../dataset/c/Rain_200_H/val',
                        help="test image dir")
    parser.add_argument("--log_dir", type=str, default='../logdir',
                        help="log_dir")
    parser.add_argument("--model_dir", type=str, default='../models',
                        help="test image dir")
    parser.add_argument("--image_size", type=int, default=256,
                        help="training patch size")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="batch size")
    parser.add_argument("--epochs", type=int, default=800,
                        help="number of epochs")
    parser.add_argument("--num_workers", type=int, default=8,
                        help="numworks in dataloader")
    parser.add_argument("--aug_data", type=bool, default=False,
                        help="whether to augment data")
    parser.add_argument("--lr", type=float, default=0.1,
                        help="learning rate")
    parser.add_argument("--loss", type=str, default="MSE",
                        help="loss; MSE', 'L1Loss', or 'MyLoss' is expected")
    parser.add_argument("--opt", type=str, default="SGD",
                        help="Optimizer for updating the network parameters")
    parser.add_argument("--checkpoint", type=str, default="the_end",
                        help="model architecture ('Similarity')")
    parser.add_argument("--sessname", type=str, default="ERL_baseline_withoutREN",
                        help="different session names for parameter modification")
    args = parser.parse_args()

    return args


def ensure_dir(dir_path):
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)


class Session:
    def __init__(self, args):
        self.log_dir = args.log_dir
        self.model_dir = args.model_dir
        ensure_dir(self.log_dir)
        ensure_dir(self.model_dir)
        logger.info('set log dir as %s' % self.log_dir)
        logger.info('set model dir as %s' % self.model_dir)

        self.net = ERLNet(in_channels=3, out_channels=3).cuda()
        self.ssim = SSIM().cuda()
        self.image_size = args.image_size
        self.batch_size = args.batch_size
        self.num_workers = args.num_workers

        self.step = 0
        self.epoch = args.epochs
        self.now_epoch = 0
        self.start_epoch = 0
        self.writers = {}
        self.total_step = 0

        self.sessname = args.sessname

        # if args.loss == "MSE":
        #     self.crit = MSELoss().cuda()
        # elif args.loss == "L1Loss":
        #     self.crit = L1Loss().cuda()
        # else:
        #     self.crit = Myloss().cuda()
        #
        # if args.opt == "SGD":
        #     self.opt = SGD(self.net.parameters(), lr=args.lr)
        # else:
        #     self.opt = Adam(self.net.parameters(), lr=args.lr)
        #
        # self.sche = MultiStepLR(self.opt, milestones=[100, 200, 300, 400, 500, 600, 700], gamma=0.5)

    def tensorboard(self, name):
        path = os.path.join(self.log_dir, self.sessname)
        ensure_dir(path)
        self.writers[name] = SummaryWriter(os.path.join(path, name + '.events'))
        return self.writers[name]

    def write(self, name, loss, ssim, epoch, image_last_train, image_val):
        lr = self.opt.param_groups[0]['lr']
        self.writers[name].add_scalar("lr", lr, epoch)
        self.writers[name].add_scalars("loss", {"train": loss[0], "test": loss[1]}, epoch)
        self.writers[name].add_scalars("ssim", {"train": ssim[0], "test": ssim[1]}, epoch)
        # self.writers[name].add_image("train result", make_grid(image_last_train))#####
        # self.writers[name].add_image("val result", image_val[0, :, :, :])#########

    def write_close(self, name):
        self.writers[name].close()

    def get_dataloader(self, dir, name):
        if name == "train":
            dataset = TrainDataset(dir, self.image_size, aug_data=args.aug_data)
            a = DataLoader(dataset, batch_size=self.batch_size, shuffle=True,
                           num_workers=self.num_workers, drop_last=True)
            self.total_step = len(a)
            return a
        elif name == "val":
            dataset = TestDataset(dir, self.image_size)
            return DataLoader(dataset, batch_size=1, shuffle=False,
                              num_workers=self.num_workers, drop_last=True)
        else:
            print("Incorrect Name for Dataloader!!!")
            return 0

    def save_checkpoints(self, name):
        dir = os.path.join(self.model_dir, self.sessname)
        ensure_dir(dir)
        ckp_path = os.path.join(dir, name)
        obj = {
            'net': self.net.state_dict(),
            'now_epoch': self.now_epoch + 1,
            'opt': self.opt.state_dict(),
        }
        torch.save(obj, ckp_path)

    def load_checkpoints(self, dir):
        ckp_path = dir
        try:
            obj = torch.load(ckp_path)
            logger.info('Load checkpoint %s' % ckp_path)
        except FileNotFoundError:
            logger.info('No checkpoint %s!!' % ckp_path)
            return
        self.net.load_state_dict(obj['net'])
        self.opt.load_state_dict(obj['opt'])
        self.start_epoch = obj['now_epoch']

    # def inf_batch(self, name, batch):
    #     OB, GT = batch['OB'].cuda(), batch['GT'].cuda()
    #     derain = self.net(OB)
    #     loss = self.crit(derain, GT)
    #     ssim = self.ssim(derain, GT)
    #     if name == 'train':
    #         self.net.zero_grad()
    #         loss.backward()
    #         self.opt.step()
    #         lr_now = self.opt.param_groups[0]["lr"]
    #         logger.info("epoch %d/%d: step %d/%d: loss is %f ssim is %f lr is %f"
    #                     % (self.now_epoch, self.epoch, self.step, self.total_step, loss, ssim, lr_now))
    #         self.step += 1
    #
    #     return derain, loss.item(), ssim.item()
    # def test_batch(self, batch):

    def save_image(self, name, img_lists):
        data, pred, label = img_lists
        pred = pred.cpu().data

        data, label, pred = data * 255, label * 255, pred * 255
        pred = np.clip(pred, 0, 255)

        h, w = pred.shape[-2:]

        gen_num = (6, 2)
        img = np.zeros((gen_num[0] * h, gen_num[1] * 3 * w, 3))
        for img_list in img_lists:
            for i in range(gen_num[0]):
                row = i * h
                for j in range(gen_num[1]):
                    idx = i * gen_num[1] + j
                    tmp_list = [data[idx], pred[idx], label[idx]]
                    for k in range(3):
                        col = (j * 3 + k) * w
                        tmp = np.transpose(tmp_list[k], (1, 2, 0))
                        img[row: row + h, col: col + w] = tmp

        img_file = os.path.join(self.log_dir, '%d_%s.jpg' % (self.step, name))
        cv2.imwrite(img_file, img)

    def epoch_out(self):
        self.step = 0


def run_train_val(args):
    sess = Session(args)
    # sess.load_checkpoints("../models/epoch 6_ssim 0.672994")
    sess.tensorboard('rain200H')
    ssim_m = 0.0
    sess.now_epoch = sess.start_epoch
    for epoch in range(int(sess.epoch - sess.start_epoch)):
        epoch = epoch + sess.start_epoch
        dt_train = sess.get_dataloader(dir=args.train_dir, name='train')
        dt_val = sess.get_dataloader(dir=args.test_dir, name='val')
        sess.net.train()
        loss_train = []
        ssim_train = []
        for batch in dt_train:
            sess.net.set_input(batch)
            result_train, loss, ssim = sess.net.optimize_parameters1()
            loss_train.append(loss)
            ssim_train.append(ssim)
        sess.epoch_out()
        loss_test = []
        ssim_test = []
        sess.net.eval()
        for batch in dt_val:
            result_val, loss, ssim = sess.test_batch("val", batch)
            loss_test.append(loss)
            ssim_test.append(ssim)
        sess.write(name="rain200H", loss=[np.mean(loss_train), np.mean(loss_test)],
                   ssim=[np.mean(ssim_train), np.mean(ssim_test)], epoch=epoch, image_last_train=result_train,
                   image_val=result_val)
        if np.mean(ssim_test) > ssim_m:
            logger.info('ssim increase from %f to %f now' % (ssim_m, np.mean(ssim_test)))
            ssim_m = np.mean(ssim_test)
            sess.save_checkpoints("epoch %d_ssim %f " % (epoch, ssim_m))
            logger.info('save model as epoch_%d_ssim %f' % (epoch, ssim_m))
        else:
            logger.info("ssim not increase from %f" % ssim_m)
        sess.now_epoch += 1
        sess.sche.step(epoch=epoch)
    sess.write_close("rain200H")

    ###########training stage2######################3
    for epoch in range(int(sess.epoch - sess.start_epoch)):
        epoch = epoch + sess.start_epoch
        sess.net.train()
        loss_train = []
        ssim_train = []
        for batch in dt_train:
            sess.net.set_input(batch)
            result_train, loss, ssim = sess.net.optimize_parameters2()
            loss_train.append(loss)
            ssim_train.append(ssim)
        sess.epoch_out()
        loss_test = []
        ssim_test = []
        sess.net.eval()
        for batch in dt_val:
            result_val, loss, ssim = sess.test_batch("val", batch)
            loss_test.append(loss)
            ssim_test.append(ssim)
        sess.write(name="rain200H", loss=[np.mean(loss_train), np.mean(loss_test)],
                   ssim=[np.mean(ssim_train), np.mean(ssim_test)], epoch=epoch, image_last_train=result_train,
                   image_val=result_val)
        if np.mean(ssim_test) > ssim_m:
            logger.info('ssim increase from %f to %f now' % (ssim_m, np.mean(ssim_test)))
            ssim_m = np.mean(ssim_test)
            sess.save_checkpoints("epoch %d_ssim %f " % (epoch, ssim_m))
            logger.info('save model as epoch_%d_ssim %f' % (epoch, ssim_m))
        else:
            logger.info("ssim not increase from %f" % ssim_m)
        sess.now_epoch += 1
        sess.sche.step(epoch=epoch)
    sess.write_close("rain200H")

############training stage3############################
    for epoch in range(int(sess.epoch - sess.start_epoch)):
        epoch = epoch + sess.start_epoch
        sess.net.train()
        loss_train = []
        ssim_train = []
        for batch in dt_train:
            sess.net.set_input(batch)
            result_train, loss, ssim = sess.net.optimize_parameters3()
            loss_train.append(loss)
            ssim_train.append(ssim)
        sess.epoch_out()
        loss_test = []
        ssim_test = []
        sess.net.eval()
        for batch in dt_val:
            result_val, loss, ssim = sess.test_batch("val", batch)
            loss_test.append(loss)
            ssim_test.append(ssim)
        sess.write(name="rain200H", loss=[np.mean(loss_train), np.mean(loss_test)],
                   ssim=[np.mean(ssim_train), np.mean(ssim_test)], epoch=epoch, image_last_train=result_train,
                   image_val=result_val)
        if np.mean(ssim_test) > ssim_m:
            logger.info('ssim increase from %f to %f now' % (ssim_m, np.mean(ssim_test)))
            ssim_m = np.mean(ssim_test)
            sess.save_checkpoints("epoch %d_ssim %f " % (epoch, ssim_m))
            logger.info('save model as epoch_%d_ssim %f' % (epoch, ssim_m))
        else:
            logger.info("ssim not increase from %f" % ssim_m)
        sess.now_epoch += 1
        sess.sche.step(epoch=epoch)
    sess.write_close("rain200H")

if __name__ == '__main__':
    log_level = 'info'
    logger = logging.getLogger('train')
    logger.setLevel(logging.INFO)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    args = get_args()
    run_train_val(args=args)

