# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# %%
import os
import numpy as np
from PIL import Image
from glob import glob

from pycocotools.coco import COCO

from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import transforms
import lightning as L

from .ext_transforms import *
# from ext_transforms import *

__all__ = ['GhostSeg_DE',
           'GhostSeg_PS', 
           'GhostSeg_DEPS']

# %%
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

data_transforms = {
    'train': ExtCompose([
            ExtRandomScale((0.5, 2.0)),
            ExtRandomCrop(size=(500, 2000), pad_if_needed=True),
            ExtRandomHorizontalFlip(p=0.5),
            ExtRandomVerticalFlip(p=0.5),
            ExtRandomRotation(degrees=(-5, 5)),
            ExtColorJitter(brightness=0.2, contrast=0.2),
            ExtRandomGaussianBlur(kernel_size_range=(3, 7), sigma_range=(0.1, 1.5), p=0.5),
            ExtToTensor(),
            ExtNormalize(mean, std),
        ]),
    'val': ExtCompose([
            ExtResize((500, 2000)),
            ExtToTensor(),
            ExtNormalize(mean, std),
        ]),
    'pred': transforms.Compose([
        transforms.Resize((500, 2000)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
}

# %%
class GhostSegDS(Dataset):
    def __init__(self,
                 ann_coco,
                 dataset_root,
                 dset='train',
                 path_replace = 'wwf_ann/Germany/',
                 transforms=None):

        self.transforms = transforms
        self.ann_coco = ann_coco
        self.dataset_root = dataset_root
        self.img_ids = []
        self.dset = dset
        self.path_replace = path_replace
        self.load_data(dset)

    def load_data(self, dset):
        for img_id in self.ann_coco.getImgIds():
            img_info = self.ann_coco.imgs[img_id]
            if img_info['height'] > 0 and img_info['width'] > 0:
                if dset == 'train':
                    if 'train' in img_info['file_name'] or 'test' in img_info['file_name']:
                        self.img_ids.append(img_id)
                elif dset == 'val' or dset == 'test':
                    if 'val' in img_info['file_name']:
                        self.img_ids.append(img_id)

    def __getitem__(self, index):
        
        img_id = self.img_ids[index]
        img_info = self.ann_coco.imgs[img_id]
        img_anns = self.ann_coco.loadAnns(self.ann_coco.getAnnIds(imgIds=img_id))
            
        mask = np.zeros((img_info['height'], img_info['width']), dtype=np.uint8)
        for i in range(len(img_anns)):
            mask += self.ann_coco.annToMask(img_anns[i])
        mask = np.clip(mask, 0, 1, out=mask)

        file_id = os.path.join(self.dataset_root,
                               img_info['file_name'].replace(self.path_replace, ''))
        img = Image.open(file_id).convert('RGB')
        target = Image.fromarray(mask).convert('L')
        # target = Image.fromarray(mask * 255).convert('L')

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        # return img, target, file_id, [i['area'] for i in img_anns]
        return img, target, file_id

    def __len__(self):
        return len(self.img_ids)


class GhostSegDS_Pred(Dataset):
    def __init__(self,
                 pred_root,
                 extension='jpg',
                 transforms=None):

        self.transforms = transforms
        self.dataset_root = pred_root 
        self.extension = extension
        self.data = []
        self.load_data()

    def load_data(self):
        self.data = glob(os.path.join(self.dataset_root, "**/*.{}".format(self.extension)), recursive=True)

    def __getitem__(self, index):
        
        data_path = self.data[index]
        img = Image.open(data_path).convert('RGB')

        ori_size = img.size

        if self.transforms is not None:
            img = self.transforms(img)

        return img, data_path, torch.Tensor(ori_size)

    def __len__(self):
        return len(self.data)

# %%
class GhostSegDM(L.LightningDataModule):
    def __init__(self, conf, ann_path, path_replace):
        super(GhostSegDM, self).__init__()
        self.conf = conf
        self.prepare_data_per_node = True 
        self._log_hyperparams = False
        self.ann_path = ann_path
        self.path_replace = path_replace

        print("Loading data...")
        if not self.conf.predict:
            # ann_coco = COCO(os.path.join(conf.dataset_root,
            #                              'SegAnn/Germany_ABS_11122024.json'))
            ann_coco = COCO(self.ann_path)

            self.dset_tr = GhostSegDS(ann_coco=ann_coco,
                                      dataset_root=conf.dataset_root,   
                                      dset='train',
                                      path_replace=self.path_replace,
                                      transforms=data_transforms['train'])

            self.dset_te = GhostSegDS(ann_coco=ann_coco,
                                      dataset_root=conf.dataset_root,   
                                      dset='val',
                                      path_replace=self.path_replace,
                                      transforms=data_transforms['val'] )

        else:
            self.dset_pd = GhostSegDS_Pred(pred_root=conf.predict_root, 
                                           extension='jpg',
                                           transforms=data_transforms['pred'])

        print("Done.")

    def train_dataloader(self):
        return DataLoader(
            self.dset_tr, batch_size=self.conf.batch_size, shuffle=True, 
            pin_memory=True, num_workers=self.conf.num_workers, drop_last=True,
            persistent_workers=True if self.conf.num_workers > 0 else False 
        )

    def val_dataloader(self):
        return DataLoader(
            self.dset_te, batch_size=self.conf.batch_size, shuffle=False, 
            pin_memory=True, num_workers=self.conf.num_workers, drop_last=False,
            persistent_workers=True if self.conf.num_workers > 0 else False 
        )

    def test_dataloader(self):
        return DataLoader(
            self.dset_te, batch_size=self.conf.batch_size, shuffle=False, 
            pin_memory=True, num_workers=self.conf.num_workers, drop_last=False,
            persistent_workers=True if self.conf.num_workers > 0 else False 
        )

    def predict_dataloader(self):
        return DataLoader(
            self.dset_pd, batch_size=self.conf.batch_size, shuffle=False, 
            pin_memory=True, num_workers=self.conf.num_workers, drop_last=False,
            persistent_workers=True if self.conf.num_workers > 0 else False 
        )

class GhostSeg_DE(GhostSegDM):
    def __init__(self, conf):
        super(GhostSeg_DE, self).__init__(conf=conf,
                                          ann_path=os.path.join(conf.dataset_root,
                                                                'SegAnn/Germany_ABS_11122024.json'),
                                          path_replace='wwf_ann/Germany/')
        self.dataset_name = 'Germany'

class GhostSeg_PS(GhostSegDM):
    def __init__(self, conf):
        super(GhostSeg_PS, self).__init__(conf=conf,
                                          ann_path=os.path.join(conf.dataset_root,
                                                                '/SegAnn/2024_04_04_cut.json'),
                                          path_replace='')
        self.dataset_name = 'Puget Sound'


class GhostSeg_DEPS(GhostSegDM):
    def __init__(self, conf):
        super(GhostSegDM, self).__init__()
        self.conf = conf
        self.prepare_data_per_node = True 
        self._log_hyperparams = False
        self.ann_path_de = os.path.join(conf.dataset_root,
                                        'Germany/SegAnn/Germany_ABS_11122024.json')
        self.path_replace_de = 'wwf_ann/Germany/'
        self.ann_path_ps = os.path.join(conf.dataset_root,
                                        'PugetSound/2024_04_04_cut/SegAnn/2024_04_04_cut.json')
        self.path_replace_ps = '' 

        print("Loading data...")
        if not self.conf.predict:
            # ann_coco = COCO(os.path.join(conf.dataset_root,
            #                              'SegAnn/Germany_ABS_11122024.json'))
            ann_coco_de = COCO(self.ann_path_de)
            ann_coco_ps = COCO(self.ann_path_ps)

            self.dset_tr_de = GhostSegDS(ann_coco=ann_coco_de,
                                         dataset_root=os.path.join(conf.dataset_root, 'Germany'),   
                                         dset='train',
                                         path_replace=self.path_replace_de,
                                         transforms=data_transforms['train'])

            self.dset_te_de = GhostSegDS(ann_coco=ann_coco_de,
                                         dataset_root=os.path.join(conf.dataset_root, 'Germany'),   
                                         dset='val',
                                         path_replace=self.path_replace_de,
                                         transforms=data_transforms['val'] )

            self.dset_tr_ps = GhostSegDS(ann_coco=ann_coco_ps,
                                         dataset_root=os.path.join(conf.dataset_root, 'PugetSound/2024_04_04_cut'),   
                                         dset='train',
                                         path_replace=self.path_replace_ps,
                                         transforms=data_transforms['train'])

            self.dset_te_ps = GhostSegDS(ann_coco=ann_coco_ps,
                                         dataset_root=os.path.join(conf.dataset_root, 'PugetSound/2024_04_04_cut'),   
                                         dset='val',
                                         path_replace=self.path_replace_ps,
                                         transforms=data_transforms['val'] )

            self.dset_tr = ConcatDataset([self.dset_tr_de, self.dset_tr_ps])
            # self.dset_te = ConcatDataset([self.dset_te_de, self.dset_te_ps])
            self.dset_te = self.dset_te_ps

        else:
            self.dset_pd = GhostSegDS_Pred(pred_root=conf.predict_root, 
                                           extension='jpg',
                                           transforms=data_transforms['pred'])

        print("Done.")

# %%
# root_path = '/home/miao/datasets/wwf_seg/PugetSound/2024_04_04_cut'
# ann_coco = COCO(os.path.join(root_path,
#                              'SegAnn/2024_04_04_cut.json'))
# ds = GhostSegDS(ann_coco=ann_coco,
#                 dataset_root=root_path, 
#                 dset='val', 
#                 path_replace='',
#                 transforms=data_transforms['val'])

# # %%
# len(ds)

# %%
