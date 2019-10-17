from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
from PIL import Image

class DogCatDataset(Dataset):
    def __init__(self, dir, transform):
        self.transform = transform
        imgs = os.listdir(dir)
        self.imgs = [os.path.join(dir, img) for img in imgs]

    def __getitem__(self, index):
        path = self.imgs[index]
        if 'dog' in path:
            label = 0
        else:
            label = 1
        img = Image.open(path).convert('RGB')
        img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.imgs)
