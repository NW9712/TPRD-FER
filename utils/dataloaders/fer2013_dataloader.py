from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import random
import numpy as np
import pandas as pd
from PIL import Image
import json
import os
import torch
import torchvision
from torchvision.transforms import autoaugment

class fer2013_dataset(Dataset):
    def __init__(self, fer2013_path, transform, mode):
        self.fer2013_path=fer2013_path
        self.transform = transform
        self.mode = mode

        self.file_paths = []
        self.label = []
        if mode == 'train':
            root = ['train']
        else:
            root = ['test']
        for r in root:
            df = pd.read_csv(os.path.join(self.fer2013_path, r + '.csv'))
            self.file_paths+=df['pixels'].tolist()
            label_c = pd.get_dummies(df['emotion'])
            self.label+= [label_c.iloc[i].idxmax() for i in range(len(label_c))]

        # rafdb   ['surprise', 'fear', 'disgust', 'happy', 'sad', 'anger', 'neutral']
        # fer2013 ['anger', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
        self.label = [5 if x == 0
                    else 2 if x == 1
                    else 1 if x == 2
                    else 3 if x == 3
                    else 4 if x == 4
                    else 0 if x == 5
                    else 6 if x == 6
                    else 7
                    for x in self.label]

        print("%s data has a size of %d" % (self.mode, len(self.file_paths)))

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        pixels=self.file_paths[index]
        pixels=list(map(int,pixels.split(' ')))
        img=np.asarray(pixels).reshape(48,48)
        img=img.astype(np.uint8)
        img=np.dstack([img]*3)
        img=self.transform(img)
        target=self.label[index]
        return img, target



class fer2013_dataloader():
    def __init__(self, cfg):
        self.cfg = cfg
        self.fer2013_path = cfg.dataset_root
        self.batch_size = cfg.batchsize
        self.num_workers = cfg.num_workers
        self.drop_last = cfg.drop_last
        self.input_size = cfg.input_size
        assert self.input_size == 224 or 112, 'Please check your input size, only 224 or 112 are permitted.'

        self.transform_test = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((self.input_size, self.input_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),
            ])

    def run(self, mode):
        if mode == 'test':
            test_dataset = fer2013_dataset(fer2013_path=self.fer2013_path, transform=self.transform_test, mode=mode)
            test_loader = DataLoader(
                dataset=test_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory = True)
            return test_loader
        else:
            raise NotImplementedError

