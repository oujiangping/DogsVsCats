import torchvision.transforms as transforms
import torch.optim as optim
from DogCatDataset import DogCatDataset
from torch.utils.data import Dataset, DataLoader
import torch
import torchvision
import sys
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

mean=[0.485, 0.456, 0.406]
std=[0.229, 0.224, 0.225]
image_size = 224
image_rows = 5
transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),])

train_data = DogCatDataset(dir='./train', transform=transform)
train_loader = DataLoader(dataset=train_data, batch_size=128, shuffle=True, num_workers=4)


test_data = DogCatDataset(dir='./test', transform=transform)
test_loader = DataLoader(dataset=train_data, batch_size=20, shuffle=True, num_workers=4)

net = torchvision.models.resnet34()
net.fc = torch.nn.Linear(in_features=512, out_features=2, bias=True)
print(net)

#optimizer = torch.optim.SGD(list(net.parameters())[:], lr=0.001, momentum=0.9)
optimizer = torch.optim.Adam(net.parameters(), lr=0.001, betas=(0.9, 0.99))
loss_func = torch.nn.CrossEntropyLoss()

def read_one_img(path, transform=transform):
    output = []
    img = Image.open(path).convert('RGB')
    if transform is not None:
        img = transform(img)
    return img.unsqueeze(0)

def train_model():
    for epoch in range(10):
        for i, (img, label) in enumerate(train_loader):
            optimizer.zero_grad()
            out = net(img)
            loss = loss_func(out, label)
            loss.backward()
            optimizer.step()
            pre = torch.max(out, 1)[1]
            correct = pre.eq(label).sum(0).numpy()
            print(epoch, i, 'loss is {:.4f}'.format(loss.item()), 'correct is {:.2f}%'.format(correct/len(img)*100))
    torch.save(net, './net.pth')

def show_label(labels):
    labels = labels.numpy()
    index = 0
    for label in labels:
        if(label == 0):
            pre = 'dog'
        else:
            pre = 'cat'
        print(pre)
        x = (index % image_rows) * image_size + 30
        y = ((int)(index/image_rows)) * image_size + 30
        plt.text(x, y, pre, family='fantasy', fontsize=12, style='italic',color='mediumvioletred')
        index = index + 1

def test_model():
    net_trained = torch.load('./net.pth')
    net_trained.eval()
    for i, (img, label) in enumerate(test_loader):
        out = net_trained(img)
        pre = torch.max(out, 1)[1]
        print(pre)
        show_label(pre)
        image = torchvision.utils.make_grid(img, nrow=image_rows).numpy()
        image = np.transpose(image, (1,2,0))
        image = image*std + mean
        plt.imshow(image)
        plt.show()
        return


def test_one_img(img_path):
    net_trained = torch.load('./net.pth')
    net_trained.eval()

    img = read_one_img(path=img_path, transform=transform)
    out = net_trained(img)
    print(out)
    pre = torch.max(out, 1)[1]
    print(pre)
    image = torchvision.utils.make_grid(img).numpy()
    plt.imshow(np.transpose(image,(1,2,0)))
    plt.show()


if sys.argv[1] == 'train':
    train_model()
if sys.argv[1] == 'test_one':
    test_one_img(img_path=sys.argv[2])
if sys.argv[1] == 'test':
    test_model()
