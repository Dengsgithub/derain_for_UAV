import os
import cv2
import numpy as np
from numpy.random import RandomState
from torch.utils.data import Dataset


# Dataset consisting of concatenated image pairs (Ground Truth in the left and Observation in the right)
class TrainDataset(Dataset):
    def __init__(self, dir, patch_size, aug_data):
        super().__init__()
        self.rand_state = RandomState()
        self.root_dir = dir
        self.mat_files = os.listdir(self.root_dir)
        self.patch_size = patch_size
        self.aug_data = aug_data
        self.file_num = len(self.mat_files)

    def __len__(self):
        return self.file_num

    def __getitem__(self, idx):
        file_name = self.mat_files[idx % self.file_num]
        img_file = os.path.join(self.root_dir, file_name)
        img_pair = cv2.imread(img_file).astype(np.float32) / 255
        h, w, _ = img_pair.shape

        if self.aug_data:
            O, B = self.crop(img_pair, aug=True)
            O, B = self.flip(O, B)
            O, B = self.rotate(O, B)
        else:
            O, B = self.crop(img_pair, aug=False)

        O = np.transpose(O, (2, 0, 1))
        B = np.transpose(B, (2, 0, 1))
        sample = {'OB': O, 'GT': B}

        return sample

    def crop(self, img_pair, aug):
        patch_size = self.patch_size
        h, ww, c = img_pair.shape
        w = int(ww / 2)

        if aug:
            mini = - 1 / 4 * self.patch_size
            maxi =   1 / 4 * self.patch_size + 1
            p_h = patch_size + self.rand_state.randint(mini, maxi)
            p_w = patch_size + self.rand_state.randint(mini, maxi)
        else:
            p_h, p_w = patch_size, patch_size

        r = self.rand_state.randint(0, h - p_h)
        c = self.rand_state.randint(0, w - p_w)
        O = img_pair[r: r+p_h, c+w: c+p_w+w]
        B = img_pair[r: r+p_h, c: c+p_w]

        if aug:
            O = cv2.resize(O, (patch_size, patch_size))
            B = cv2.resize(B, (patch_size, patch_size))

        return O, B

    def flip(self, O, B):
        if self.rand_state.rand() > 0.5:
            O = np.flip(O, axis=1)
            B = np.flip(B, axis=1)
        return O, B

    def rotate(self, O, B):
        angle = self.rand_state.randint(-30, 30)
        patch_size = self.patch_size
        center = (int(patch_size / 2), int(patch_size / 2))
        M = cv2.getRotationMatrix2D(center, angle, 1)
        O = cv2.warpAffine(O, M, (patch_size, patch_size))
        B = cv2.warpAffine(B, M, (patch_size, patch_size))
        return O, B


class TestDataset(Dataset):
    def __init__(self, dir, patch_size):
        super().__init__()
        self.rand_state = RandomState(66)
        self.root_dir = dir
        self.mat_files = os.listdir(self.root_dir)
        self.patch_size = patch_size
        self.file_num = len(self.mat_files)

    def __len__(self):
        return self.file_num

    def __getitem__(self, idx):
        file_name = self.mat_files[idx % self.file_num]
        img_file = os.path.join(self.root_dir, file_name)
        img_pair = cv2.imread(img_file).astype(np.float32) / 255
        h, ww, c = img_pair.shape
        w = int(ww / 2)

        B = np.transpose(img_pair[:, :w, :], (2, 0, 1))
        O = np.transpose(img_pair[:, w:, :], (2, 0, 1))
        sample = {'OB': O, 'GT': B}

        return sample


class ShowDataset(Dataset):
    def __init__(self, name):
        super().__init__()
        self.rand_state = RandomState(66)
        self.root_dir = os.path.join(data_dir, name)
        self.img_files = sorted(os.listdir(self.root_dir))
        self.file_num = len(self.img_files)

    def __len__(self):
        return self.file_num

    def __getitem__(self, idx):
        file_name = self.img_files[idx % self.file_num]
        img_file = os.path.join(self.root_dir, file_name)
        img_pair = cv2.imread(img_file).astype(np.float32) / 255

        h, ww, c = img_pair.shape
        w = int(ww / 2)

        #O = np.transpose(img_pair, (2, 0, 1))
        O = np.transpose(img_pair[:, w:], (2, 0, 1))
        sample = {'O': O, 'idx': idx}

        return sample

    def get_name(self, idx):
        return self.img_files[idx % self.file_num].split('.')[0]


if __name__ == '__main__':
    dt = TestDataset('val')
    a = dt[0]
    cv2.imwrite("D:\\Desktop\\O.png", np.transpose(a['O']*255, (1, 2, 0)))
    cv2.imwrite("D:\\Desktop\\B.png", np.transpose(a['B']*255, (1, 2, 0)))

    dt = TestDataset('val')
    print('TrainValDataset')
    for i in range(10):
        smp = dt[i]
        for k, v in smp.items():
            print(k, v.shape, v.dtype, v.mean())

    print()
    dt = TestDataset('test')
    print('TestDataset')
    for i in range(10):
        smp = dt[i]
        for k, v in smp.items():
            print(k, v.shape, v.dtype, v.mean())

    print()
    print('ShowDataset')
    dt = ShowDataset('test')
    for i in range(10):
        smp = dt[i]
        for k, v in smp.items():
            print(k, v.shape, v.dtype, v.mean())
