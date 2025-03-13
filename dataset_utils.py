# dataset_utils.py

import os
import cv2
import torch
import torchvision.transforms as T
from torch.utils.data import Dataset

class SimpleFaceDataset(Dataset):
    """
    Assume strcuture:
    root_dir
    ├─ PersonA
    │   ├─ xxx.jpg
    │   └─ ...
    └─ PersonB
        ├─ yyy.jpg
        └─ ...

    transform: torchvision.transforms
    """
    def __init__(self, root_dir, transform=None):
        super().__init__()
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []

        subdirs = os.listdir(root_dir)
        class_names = sorted(subdirs)  # e.g. ['PersonA','PersonB']
        self.class_to_idx = {cls_name:i for i,cls_name in enumerate(class_names)}

        for cls_name in class_names:
            cls_dir = os.path.join(root_dir, cls_name)
            if not os.path.isdir(cls_dir):
                continue
            label = self.class_to_idx[cls_name]
            for fname in os.listdir(cls_dir):
                fpath = os.path.join(cls_dir, fname)
                if os.path.isfile(fpath):
                    self.samples.append((fpath,label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        fpath, label = self.samples[index]
        img = cv2.imread(fpath)  # BGR
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # to RGB
        if self.transform is not None:
            pil = T.ToPILImage()(img)
            img_tensor = self.transform(pil)
        else:
            # fallback
            img_tensor = torch.tensor(img).permute(2,0,1).float()/255.
        return img_tensor, label
