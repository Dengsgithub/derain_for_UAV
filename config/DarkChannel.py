import cv2
import torch
from torch import nn
import numpy as np
from skimage.morphology import disk
from skimage.filters.rank import minimum
import time


def zmMinFilterGray(src, r=7):
    '''最小值滤波，r是滤波器半径'''
    return cv2.erode(src, np.ones((2 * r + 1, 2 * r + 1)))

class DarkChannel(nn.Module):
    def __init__(self, window_size=15):
        super(DarkChannel, self).__init__()
        self.sz = window_size

    def forward(self, img):
        img = torch.min(img, dim=1, keepdim=True)[0]
        img = img.cpu().detach().numpy()
        dark = np.zeros(img.shape)
        for i in range(img.shape[0]):
            batch = img[0][0]
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (self.sz, self.sz))
            batch = cv2.erode(batch, kernel)
            batch = np.expand_dims(batch, axis=0)
            dark[i, :, :, :] = batch
        dark = torch.from_numpy(dark)
        return dark


if __name__ == "__main__":
    dc = DarkChannel()
    a = cv2.imread(
        "D:\\Desktop\\Code\\pytorch\\RESCAN-master\\dataset\\Rain_100_H\\train\\y_out\\26.png")
    a = np.transpose(a, (2, 0, 1))
    a = torch.from_numpy(np.expand_dims(a, axis=0))
    b = dc(a)
    b = b.cpu().detach().numpy()
    b = np.transpose(b[0], (1, 2, 0))
    cv2.imwrite("D:\\Desktop\\ee.png", b)
    # cv2.waitKey()
