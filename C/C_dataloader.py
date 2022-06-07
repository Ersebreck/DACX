from torch.utils.data.dataset import Dataset
import json
import numpy as np
import os
from PIL import Image

class DAXCDataset(Dataset):
    def __init__(self, data_path, anno_path, transforms=None):
        self.transforms = transforms
        with open(anno_path) as fp:
            json_data = json.loads(json.load(fp))
        samples = json_data['samples']
        self.classes = json_data['labels']
        self.imgs = []
        self.annos = []
        self.data_path = data_path
        print('loading', anno_path)
        for sample in samples:
            self.imgs.append(sample['image_name'])
            self.annos.append(sample['image_labels'])
        for item_id in range(len(self.annos)):
            item = self.annos[item_id]
            vector = [cls in item for cls in self.classes]
            self.annos[item_id] = np.array(vector, dtype=float)

    def __getitem__(self, item):
        anno = self.annos[item]
        img_path = os.path.join(self.imgs[item])
        img = Image.open(self.data_path+img_path).convert('RGB')
        if self.transforms is not None:
            img = self.transforms(img)
        return img, anno

    def __len__(self):
        return len(self.imgs)