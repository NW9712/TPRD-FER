import os
import random
import torch
import torch.utils.data as data
import collections
import pandas as pd
import logging
import numpy as np
import csv
from PIL import Image
from torchvision import transforms
from itertools import islice
from torchvision.transforms import autoaugment

class ferplus_dataset(data.Dataset):
    def __init__(self, root_path, mode, transform = None,occlusion=False,headpose=False):
        self.mode = mode
        self.transform = transform
        self.root_path = root_path
        self.num_classes = 8
        self.occlusion = occlusion
        self.headpose = headpose

        # df = pd.read_csv(os.path.join(self.raf_path, 'EmoLabel/list_patition_label.txt'), sep=' ', header=None, names=['name' ,'label'])

        if mode == 'train':
            self.data_dir = os.path.join(self.root_path, 'FER2013Train')
            self.label_path = os.path.join(self.root_path, 'train_val.txt')
        else:
            self.data_dir = os.path.join(self.root_path, 'FER2013Test')
            if self.occlusion:
                self.label_path = os.path.join(self.root_path, 'jianfei_occlusion_list.txt')
            elif self.headpose:
                if self.headpose == 30:
                    self.label_path = os.path.join(self.root_path, 'pose_30_ferplus_list.txt')
                elif self.headpose == 45:
                    self.label_path = os.path.join(self.root_path, 'pose_45_ferplus_list.txt')
                else:
                    raise  NotImplementedError
            else:
                self.label_path = os.path.join(self.root_path, 'test.txt')

        self.file_paths = []
        self.label = []

        if mode == 'train':
            with open(self.label_path, 'r') as csvfile:
                ferplus_rows = csv.reader(csvfile, delimiter=',')
                for row in islice(ferplus_rows, 0, None):
                    onehot_label = row[0].split(' ')[0].split('/')[0]
                    img_path = row[0].split(' ')[0].split('/')[1]
                    self.label.append(int(onehot_label))
                    self.file_paths.append(os.path.join(self.data_dir, img_path + '.png'))
        else:
            if self.occlusion:
                with open(self.label_path, 'r') as csvfile:
                    ferplus_rows = csv.reader(csvfile, delimiter=',')
                    for row in islice(ferplus_rows, 0, None):
                        onehot_label = row[0].split(' ')[0].split('_')[0]
                        img_path = row[0].split(' ')[0].split('_')[1]
                        self.label.append(int(onehot_label))
                        self.file_paths.append(os.path.join(self.data_dir, img_path))
            elif self.headpose:
                with open(self.label_path, 'r') as csvfile:
                    ferplus_rows = csv.reader(csvfile, delimiter=',')
                    for row in islice(ferplus_rows, 0, None):
                        onehot_label = row[0].split('.')[0].split('/')[0]
                        img_path = row[0].split('.')[0].split('/')[1]
                        self.label.append(int(onehot_label))
                        self.file_paths.append(os.path.join(self.data_dir, img_path + '.png'))
            else:
                with open(self.label_path, 'r') as csvfile:
                    ferplus_rows = csv.reader(csvfile, delimiter=',')
                    for row in islice(ferplus_rows, 0, None):
                        onehot_label = row[0].split(' ')[0].split('/')[0]
                        img_path = row[0].split(' ')[0].split('/')[1]
                        self.label.append(int(onehot_label))
                        self.file_paths.append(os.path.join(self.data_dir, img_path + '.png'))

        self.label = [6 if x == 0
                    else 3 if x == 1
                    else 0 if x == 2
                    else 4 if x == 3
                    else 5 if x == 4
                    else 2 if x == 5
                    else 1 if x == 6
                    else 7
                    for x in self.label]
        print(collections.Counter(self.label))
        print("%s data has a size of %d" % (self.mode, len(self.label)))
#
    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        image = Image.open(path).convert('RGB')
        label = self.label[idx]
        image = self.transform(image)
        # return image, torch.FloatTensor(label)
        return image, label


class ferplus_dataloader():
    def __init__(self, cfg):
        self.cfg = cfg
        self.root_path=cfg.dataset_root
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
                                      std=[0.229, 0.224, 0.225]),
                 ])
            test_dataset = ferplus_dataset(root_path = self.root_path, mode = mode, transform = transforms_test,occlusion=self.cfg.occlusion, headpose=self.cfg.headpose)
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
