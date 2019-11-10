import os.path as osp
import torch
import torch.utils.data as data
import data.util as util
from PIL import Image
import numpy as np
from . import degradation

class ReferenceRestorationDataset(data.Dataset):
    """
    A dataset for reference based image restoration
    """

    def __init__(self, opt):
        super(ReferenceRestorationDataset, self).__init__()
        self.opt = opt
        
        self.root = opt['dataroot_GT']
        # self.data_info = {'path_LQ': [], 'path_GT': [], 'folder': [], 'idx': [], 'border': []}
        self.disparity = self.opt['disparity']
        self.img_folder = self.root+"disparity{}/".format(self.disparity)

        with open(self.root+"disparity{}.txt".format(self.disparity),'r') as f:
            im_files = f.read()
        im_files = im_files.split('\n')
        if len(im_files[-1]) == 0:
            im_files = im_files[:-1]
        self.pairs = [p.split(" ") for p in im_files]
        assert len(self.pairs[0]) == 2, "Error in parsing imfiles."
        print("Found {} image pairs".format(len(self.pairs)))

        if (self.opt["phase"] != "train"):
            self.pairs = self.pairs[::350]
        
        self.degrade_func = getattr(degradation, self.opt['distortion'])
        self.ref_degrade_func = degradation.exposure

        # for i in range(max_idx):
        #     self.data_info['idx'].append('{}/{}'.format(i, max_idx))
        # border_l = [0] * max_idx
        # for i in range(1):
        #     border_l[i] = 1
        #     border_l[max_idx - i - 1] = 1
        # self.data_info['border'].extend(border_l)

    def __getitem__(self, index):
        path1 = self.img_folder+self.pairs[index][0]
        path2 = self.img_folder+self.pairs[index][1]


        with open(path1, 'rb') as f:
            img1 = Image.open(f)
            img1 = np.array(img1.convert('RGB').resize((512,288))).astype(np.float32)
        with open(path2, 'rb') as f:
            img2 = Image.open(f)
            img2 = np.array(img2.convert('RGB').resize((512, 288))).astype(np.float32)

        img1 = img1/255.0
        img2 = img2/255.0
        
        img1_degraded = np.asarray(self.degrade_func(img1), 'float32')
        ref = np.asarray(self.ref_degrade_func(img2), 'floatt32')


        if (self.opt["phase"] == "train") and self.opt['use_flip']:
            ref, img1, img1_degraded = util.augment180([ref,img1,img1_degraded])

        img_LQ = np.stack([ref,img1_degraded],axis=0)
        img_GT = img1
        img_LQ = torch.from_numpy(np.ascontiguousarray(img_LQ)).permute(0,3,1,2)
        img_GT = torch.from_numpy(np.ascontiguousarray(img_GT)).permute(2,0,1)
        return {'LQs': img_LQ, 'GT': img_GT, 'folder':"main", 'idx':index}

    def __len__(self):
        return len(self.pairs)


