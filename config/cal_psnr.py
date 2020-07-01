from torch import nn
import torch

class PSNR(nn.Module):
    def __init__(self, ):
        super(PSNR, self).__init__()

    def forward(self, img1, img2):
        return _ssim(img1, img2, window, self.window_size, channel, self.size_average)

    def psnr_cal(self, x, y):
        x_t = x *255.0
        max_pixel = 255.0
        y_t = K.clip(y*255.0, 0.0, 255.0)
        # y = K.clip(y, 0.0, 1.0)
        return 10.0 * torch.log10((max_pixel ** 2) / (torch.mean(torch.pow((y_t - x_t), 2))))