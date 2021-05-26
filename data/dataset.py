import os
import numpy as np


from torch.utils import data
from torchvision import transforms as T
from PIL import Image

class DogCat(data.Dataset):
    """Dataset主要用于封装数据，多线程加载"""

    def __init__(self,root,transforms=None,train=True,test=False):
        '''
        func: 获取所有图片地址，并根据训练，验证，测试划分数据
        :param root: 地址
        :param transforms: 是否变换
        :param train: 训练模式
        :param test: 测试模式
        '''

        # 导入数据
        self.test = test
        imgs = [os.path.join(root, img) for img in os.listdir(root)]

        # test1: data/test1/8973.jpg
        # train: data/train/cat.10004.jpg
        if self.test:
            imgs = sorted(imgs, key=lambda x: int(x.split('.')[-2].split('/')[-1])) # 按照cat排序
        else:
            imgs = sorted(imgs, key=lambda x: int(x.split('.')[-2]))

        imgs_num = len(imgs)

        # shuffle imgs
        imgs = np.random.permutation(imgs)

        # 数据集划分，验证:训练 3:7
        if self.test:
            self.imgs = imgs # 测试使用全量数据混序
        elif train:
            self.imgs = imgs[:int(0.7 * imgs_num)]
        else:
            self.imgs = imgs[int(0.7 * imgs_num):] # val

        # 数据预处理，可以提前写入dataprocessing文件
        if transforms is None:
            normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
            # 测试和验证集
            if self.test or not train:
                self.transforms = T.Compose([
                    T.Resize(224),
                    T.RandomResizedCrop(224),
                    T.ToTensor(),
                    normalize
                ])
            else: # 训练集
                self.transforms = T.Compose([
                    T.Resize(256),
                    T.RandomResizedCrop(224),
                    T.RandomHorizontalFlip(),
                    T.ToTensor(),
                    normalize
                ])

    def __getitem__(self, item):
        """
        func: 一次返回一张图片的数据，测试集没有label则返回图片ID,如1000.jpg返回1000
        :param item:
        :return:
        """
        img_path = self.imgs[item]
        if self.test:
            label = int(self.imgs[item].split('.')[-2].split('/')[-1])
        else:
            label = 1 if 'dog' in img_path.split('/')[-1] else 0
        data = Image.open(img_path)
        data = self.transforms(data)
        return data, label

    def __len__(self):
        """
        func:数据集中所有图片的数量
        :return:
        """
        return len(self.imgs)


if __name__ == "__main__":
    # 这里书写单元测试
    # train_dataset = DogCat(opt.train_data_root, train=True)
    # trainloader = DataLoader(train_dataset,
    #                          batch_size=opt.batch_size,
    #                          shuffle=True,
    #                          num_workers=opt.num_workers)
    #
    # for ii, (data, label) in enumerate(trainloader):
    #     train()
    pass

