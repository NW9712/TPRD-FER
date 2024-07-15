import os
import torch
import torch.utils.data as data
import pandas as pd
import logging
import torchvision
from PIL import Image
from torchvision import transforms
import csv
from itertools import islice
import collections
import numpy as np

class affect_dataset(data.Dataset):
    def __init__(self, affect_path, mode, num_class, transform = None,occlusion=False,headpose=False):
        self.affect_path = affect_path
        self.mode = mode
        self.transform = transform
        self.num_class = num_class
        self.occlusion = occlusion
        self.headpose = headpose

        # 0:Surprise, 1:Fear, 2:Disgust, 3:Happiness, 4:Sadness, 5:Anger, 6:Neutral
        if mode == 'test':
            df = pd.read_csv(os.path.join(self.affect_path, 'validation.csv'))
        else:
            df = pd.read_csv(os.path.join(self.affect_path, 'training.csv'))

        self.data = df[df['expression'] < self.num_class]
        self.file_paths = self.data.loc[:, 'subDirectory_filePath'].values
        self.faceX = self.data.loc[:, 'face_x'].values
        self.faceY = self.data.loc[:, 'face_y'].values
        self.faceW = self.data.loc[:, 'face_width'].values
        self.faceH = self.data.loc[:, 'face_height'].values
        self.label = self.data.loc[:, 'expression'].values
        self.label = [6 if x == 0
                      else 3 if x == 1
        else 4 if x == 2
        else 0 if x == 3
        else 1 if x == 4
        else 2 if x == 5
        else 5 if x == 6
        else 7 if x == 7
        else 8
                      for x in self.label]  ##to rafdb
        self.label = np.array(self.label)

        if self.occlusion:
            occlusion_imgpath = []
            occlusion_index = []
            with open(os.path.join(self.affect_path, 'occlusion_affectnet_list.txt'), 'r') as csvfile:
                affect_rows = csv.reader(csvfile, delimiter=',')
                for row in islice(affect_rows, 0, None):
                    occlusion_imgpath.append(row[0].split('/')[2])
            file_paths_c = [filepath.split('/')[-1].split('.')[0] for filepath in self.file_paths]
            for p in occlusion_imgpath:
                if p in file_paths_c:
                    occlusion_index.append(file_paths_c.index(p))
            self.file_paths = self.file_paths[occlusion_index]
            self.faceX = self.faceX[occlusion_index]
            self.faceY = self.faceY[occlusion_index]
            self.faceW = self.faceW[occlusion_index]
            self.faceH = self.faceH[occlusion_index]
            self.label = self.label[occlusion_index]
        elif self.headpose:
            headpose_imgpath = []
            headpose_index = []
            if self.headpose == 30:
                headpose_listname='pose_30_affectnet_list.txt'
            elif self.headpose == 45:
                headpose_listname = 'pose_45_affectnet_list.txt'
            else:
                raise NotImplementedError
            with open(os.path.join(self.affect_path, headpose_listname), 'r') as csvfile:
                affect_rows = csv.reader(csvfile, delimiter=',')
                for row in islice(affect_rows, 0, None):
                    headpose_imgpath.append(row[0].split('/')[1].split('.')[0])
            file_paths_c = [filepath.split('/')[-1].split('.')[0] for filepath in self.file_paths]
            for p in headpose_imgpath:
                if p in file_paths_c:
                    headpose_index.append(file_paths_c.index(p))
            self.file_paths = self.file_paths[headpose_index]
            self.faceX = self.faceX[headpose_index]
            self.faceY = self.faceY[headpose_index]
            self.faceW = self.faceW[headpose_index]
            self.faceH = self.faceH[headpose_index]
            self.label = self.label[headpose_index]
        else:
            pass

        print(collections.Counter(self.label))
        print("%s data has a size of %d" % (self.mode, len(self.label)))

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        img, target = Image.open(os.path.join(self.affect_path, self.file_paths[index])).convert('RGB'), self.label[index]
        img = img.crop([int(self.faceX[index]), int(self.faceY[index]), int(self.faceX[index] + self.faceW[index]),
                        int(self.faceY[index] + self.faceH[index])])
        if self.transform is not None:
            img = self.transform(img)
        return img, target


class affect_dataloader():
    def __init__(self, cfg):
        self.cfg = cfg
        self.affect_path=cfg.dataset_root
        self.batchsize=cfg.batchsize
        self.num_workers=cfg.num_workers
        self.drop_last=cfg.drop_last
        self.input_size=cfg.input_size
        assert self.input_size == 224 or 112, 'Please check your input size, only 224 or 112 are permitted.'

    def run(self, mode):
        if mode == 'test':
            transforms_test = transforms.Compose(
                [transforms.Resize((self.input_size, self.input_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])])
            test_dataset = affect_dataset(affect_path = self.affect_path, mode = mode,num_class = len(self.cfg.expression_prompts),transform = transforms_test,occlusion=self.cfg.occlusion, headpose=self.cfg.headpose)
            testloader = data.DataLoader(
                dataset=test_dataset,
                batch_size=self.batchsize,
                num_workers=self.num_workers,
                pin_memory = True)
            return testloader
        else:
            raise NotImplementedError

class ImbalancedDatasetSampler(data.sampler.Sampler):
    def __init__(self, dataset, indices: list = None, num_samples: int = None):
        self.indices = list(range(len(dataset))) if indices is None else indices
        self.num_samples = len(self.indices) if num_samples is None else num_samples

        df = pd.DataFrame()
        df["label"] = self._get_labels(dataset)
        df.index = self.indices
        df = df.sort_index()

        label_to_count = df["label"].value_counts()

        weights = 1.0 / label_to_count[df["label"]]

        self.weights = torch.DoubleTensor(weights.to_list())

        # self.weights = self.weights.clamp(min=1e-5)

    def _get_labels(self, dataset):
        if isinstance(dataset, torch.utils.data.Dataset):
            return [dataset.label[i] for i in self.indices]
        else:
            raise NotImplementedError

    def __iter__(self):
        return (self.indices[i] for i in torch.multinomial(self.weights, self.num_samples, replacement=True))

    def __len__(self):
        return self.num_samples


