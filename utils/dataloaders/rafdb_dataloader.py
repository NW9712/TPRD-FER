import os
import random
import torch
import torch.utils.data as data
from torchvision import transforms
import pandas as pd
import logging
import numpy as np
from PIL import Image
import collections

class rafdb_dataset(data.Dataset):
    def __init__(self, raf_path, mode, transform = None,occlusion=False,headpose=False):
        self.mode = mode
        self.transform = transform
        self.raf_path = raf_path
        self.noisy = False
        self.occlusion = occlusion
        self.headpose = headpose

        if mode == 'train':
            if self.noisy:
                print('loading noisy label')
                df = pd.read_csv(os.path.join(self.raf_path, 'EmoLabel/noise03.txt'), sep=' ', header=None,names=['name', 'label'])
            else:
                df = pd.read_csv(os.path.join(self.raf_path, 'EmoLabel/list_patition_label.txt'), sep=' ', header=None, names=['name', 'label'])
        else:
            if self.occlusion:
                df = pd.read_csv(os.path.join(self.raf_path, 'EmoLabel/rafdb_occlusion_list.txt'), sep=' ', header=None,names=['name', 'label', 'occlusion_type'])
            elif self.headpose:
                if self.headpose == 30:
                    df = pd.read_csv(os.path.join(self.raf_path, 'EmoLabel/val_raf_db_list.txt'), sep='/',header=None,names=['label', 'name'])
                elif self.headpose == 45:
                    df = pd.read_csv(os.path.join(self.raf_path, 'EmoLabel/val_raf_db_list_45.txt'), sep='/',header=None, names=['label', 'name'])
                else:
                    raise  NotImplementedError
            else:
                df = pd.read_csv(os.path.join(self.raf_path, 'EmoLabel/list_patition_label.txt'), sep=' ', header=None, names=['name' ,'label'])

        if mode == 'train':
            self.data = df[df['name'].str.startswith('train')]
        else:
            self.data = df[df['name'].str.startswith('test')]

        if mode == 'train':
            file_names = self.data.loc[:, 'name'].values
            self.label = self.data.loc[:,'label'].values - 1  # 0:Surprise, 1:Fear, 2:Disgust, 3:Happiness, 4:Sadness, 5:Anger, 6:Neutral
        else:
            if self.occlusion:
                file_names = self.data.loc[:, 'name'].values
                self.label = self.data.loc[:,'label'].values  # 0:Surprise, 1:Fear, 2:Disgust, 3:Happiness, 4:Sadness, 5:Anger, 6:Neutral
            elif self.headpose:
                file_names = self.data.loc[:, 'name'].values
                self.label = self.data.loc[:, 'label'].values  # 0:Surprise, 1:Fear, 2:Disgust, 3:Happiness, 4:Sadness, 5:Anger, 6:Neutral
            else:
                file_names = self.data.loc[:, 'name'].values
                self.label = self.data.loc[:, 'label'].values - 1  # 0:Surprise, 1:Fear, 2:Disgust, 3:Happiness, 4:Sadness, 5:Anger, 6:Neutral

        self.file_paths = []
        for f in file_names:
            path = os.path.join(self.raf_path, 'Image/aligned_224', f)
            self.file_paths.append(path)
        # for f in file_names:
        #     f = f.split(".")[0]
        #     f = f +"_aligned.jpg"
        #     path = os.path.join(self.raf_path, 'Image/aligned', f)
        #     self.file_paths.append(path)

        print(collections.Counter(self.label))
        print("%s data has a size of %d" % (self.mode, len(self.label)))

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        path = path.replace('_aligned','')
        if not path.endswith('jpg'):
            path = path + '.jpg'
        image = Image.open(path).convert('RGB')
        label = self.label[idx]

        if self.transform is not None:
            image = self.transform(image)
            pass
        return image, label

class rafdb_dataloader():
    def __init__(self, cfg):
        self.cfg = cfg
        self.raf_path=cfg.dataset_root
        self.batchsize=cfg.batchsize
        self.num_workers=cfg.num_workers
        self.drop_last=cfg.drop_last
        self.input_size=cfg.input_size
        assert self.input_size == 224, 'Please check your input size, only 224 is permitted.'

    def run(self, mode):
        if mode == 'test':
            transforms_test = transforms.Compose(
                [transforms.Resize((self.input_size, self.input_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
                 # transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                 #                      std=[0.26862954, 0.26130258, 0.27577711]),
                 ])
            test_dataset = rafdb_dataset(raf_path = self.raf_path, mode = mode,transform = transforms_test, occlusion=self.cfg.occlusion, headpose=self.cfg.headpose)
            testloader = data.DataLoader(
                dataset=test_dataset,
                batch_size=self.batchsize,
                num_workers=self.num_workers,
                pin_memory = True)
            return testloader
        else:
            raise NotImplementedError


