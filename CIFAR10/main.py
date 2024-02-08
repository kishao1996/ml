# ref: https://zhuanlan.zhihu.com/p/25572330
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from config import DATA_DIR

def main():
    torch.multiprocessing.freeze_support()

    # torchvision数据集的输出是在[0, 1]范围内的PILImage图片。
    # compose是一批transform的组合
    transform = transforms.Compose([
        transforms.ToTensor(),
        # 按平均值0.5做标准化
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    # 加载数据集
    trainset = torchvision.datasets.CIFAR10(root=DATA_DIR, train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, 
                                            shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root=DATA_DIR, train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=4, 
                                            shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # functions to show an image

    def imshow(img):
        img = img / 2 + 0.5 # unnormalize
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1,2,0)))

    # show some random training images
    dataiter = iter(trainloader)
    images, labels = dataiter.next()

    # print images
    imshow(torchvision.utils.make_grid(images))
    # print labels
    print(' '.join('%5s'%classes[labels[j]] for j in range(4)))

if __name__ == '__main__':
    main()