import torchvision.transforms as transforms
from DogCatDataset import DogCatDataset
from DCGAN import Generator, Discriminator
from torch.utils.data import DataLoader
import torch
import torchvision
import torchvision.utils as vutils
import torch.optim as optim
import sys
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

mean=[0.485, 0.456, 0.406]
std=[0.229, 0.224, 0.225]
image_size = 64
image_rows = 5
batch_size = 64
nz = 100
transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),])

train_data = DogCatDataset(dir='./train', transform=transform)
train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)


test_data = DogCatDataset(dir='./test', transform=transform)
test_loader = DataLoader(dataset=train_data, batch_size=20, shuffle=True)

net = torchvision.models.resnet50()
net.fc = torch.nn.Linear(in_features=2048, out_features=2, bias=True)

#optimizer = torch.optim.SGD(list(net.parameters())[:], lr=0.001, momentum=0.9)
optimizer = torch.optim.Adam(net.parameters(), lr=0.001, betas=(0.9, 0.99))
criterion = torch.nn.CrossEntropyLoss()
criterion = torch.nn.BCELoss()

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

netG = Generator()
netG.apply(weights_init)
netD = Discriminator()
netD.apply(weights_init)
print(netG)
print(netD)

optimizerD = optim.Adam(netD.parameters(), lr=0.0001, betas=(0.9, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=0.0001, betas=(0.9, 0.999))

real_label = 1
fake_label = 0
fixed_noise = torch.randn(batch_size, nz, 1, 1)

for epoch in range(100):
    for i, data in enumerate(train_loader, 0):
        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        # train with real
        netD.zero_grad()
        real_cpu = data[0]
        batch_size = real_cpu.size(0)
        label = torch.full((batch_size,), real_label)

        output = netD(real_cpu)
        errD_real = criterion(output, label)
        errD_real.backward()
        D_x = output.mean().item()

        # train with fake
        noise = torch.randn(batch_size, nz, 1, 1)
        fake = netG(noise)
        label.fill_(fake_label)
        output = netD(fake.detach())
        errD_fake = criterion(output, label)
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        errD = errD_real + errD_fake
        optimizerD.step()

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        netG.zero_grad()
        label.fill_(real_label)  # fake labels are real for generator cost
        output = netD(fake)
        errG = criterion(output, label)
        errG.backward()
        D_G_z2 = output.mean().item()
        optimizerG.step()

        print('[%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
              % (epoch, i, len(train_loader),
                 errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

        if i % 100 == 0:
            vutils.save_image(real_cpu,'gan/real_samples_%d_%d.png' % (epoch, i), normalize=True)
            fake = netG(fixed_noise)
            vutils.save_image(fake.detach(), 'gan/fake_samples_epoch_%d_%d.png' % (epoch, i),normalize=True)

    # do checkpointing
    torch.save(netG.state_dict(), 'netG_epoch_%d.pth' % (epoch))
    torch.save(netD.state_dict(), 'netD_epoch_%d.pth' % (epoch))
